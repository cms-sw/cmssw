#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"

#include "SimGeneral/GFlash/interface/GflashAntiProtonShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashHistogram.h"
#include "SimGeneral/GFlash/interface/GflashHit.h"
#include "SimGeneral/GFlash/interface/GflashKaonMinusShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashKaonPlusShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashNameSpace.h"
#include "SimGeneral/GFlash/interface/GflashPiKShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashProtonShowerProfile.h"

#include "G4EventManager.hh"
#include "G4FastSimulationManager.hh"
#include "G4TouchableHandle.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VSensitiveDetector.hh"

#include "G4AntiProton.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4Proton.hh"
#include "G4VProcess.hh"

#include <vector>

using namespace CLHEP;

GflashHadronShowerModel::GflashHadronShowerModel(G4String modelName,
                                                 G4Region *envelope,
                                                 const edm::ParameterSet &parSet)
    : G4VFastSimulationModel(modelName, envelope), theParSet(parSet) {
  theWatcherOn = parSet.getParameter<bool>("watcherOn");

  theProfile = new GflashHadronShowerProfile(parSet);
  thePiKProfile = new GflashPiKShowerProfile(parSet);
  theKaonPlusProfile = new GflashKaonPlusShowerProfile(parSet);
  theKaonMinusProfile = new GflashKaonMinusShowerProfile(parSet);
  theProtonProfile = new GflashProtonShowerProfile(parSet);
  theAntiProtonProfile = new GflashAntiProtonShowerProfile(parSet);
  theHisto = GflashHistogram::instance();

  theGflashStep = new G4Step();
  theGflashTouchableHandle = new G4TouchableHistory();
  theGflashNavigator = new G4Navigator();
}

GflashHadronShowerModel::~GflashHadronShowerModel() {
  if (theProfile)
    delete theProfile;
  if (theGflashStep)
    delete theGflashStep;
}

G4bool GflashHadronShowerModel::IsApplicable(const G4ParticleDefinition &particleType) {
  return &particleType == G4PionMinus::PionMinusDefinition() || &particleType == G4PionPlus::PionPlusDefinition() ||
         &particleType == G4KaonMinus::KaonMinusDefinition() || &particleType == G4KaonPlus::KaonPlusDefinition() ||
         &particleType == G4AntiProton::AntiProtonDefinition() || &particleType == G4Proton::ProtonDefinition();
}

G4bool GflashHadronShowerModel::ModelTrigger(const G4FastTrack &fastTrack) {
  // ModelTrigger returns false for Gflash Hadronic Shower Model if it is not
  // tested from the corresponding wrapper process, GflashHadronWrapperProcess.
  // Temporarily the track status is set to fPostponeToNextEvent at the wrapper
  // process before ModelTrigger is really tested for the second time through
  // PostStepGPIL of enviaged parameterization processes.  The better
  // implmentation may be using via G4VUserTrackInformation of each track, which
  // requires to modify a geant source code of stepping (G4SteppingManager2)

  G4bool trigger = false;

  // mininum energy cutoff to parameterize
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() / GeV < Gflash::energyCutOff)
    return trigger;

  // check whether this is called from the normal GPIL or the wrapper process
  // GPIL
  if (fastTrack.GetPrimaryTrack()->GetTrackStatus() == fPostponeToNextEvent) {
    // Shower pameterization start at the first inelastic interaction point
    G4bool isInelastic = isFirstInelasticInteraction(fastTrack);

    // Other conditions
    if (isInelastic) {
      trigger = (!excludeDetectorRegion(fastTrack));
    }
  }

  return trigger;
}

void GflashHadronShowerModel::DoIt(const G4FastTrack &fastTrack, G4FastStep &fastStep) {
  // kill the particle
  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);

  // parameterize energy depostion by the particle type
  G4ParticleDefinition *particleType = fastTrack.GetPrimaryTrack()->GetDefinition();

  theProfile = thePiKProfile;
  if (particleType == G4KaonMinus::KaonMinusDefinition())
    theProfile = theKaonMinusProfile;
  else if (particleType == G4KaonPlus::KaonPlusDefinition())
    theProfile = theKaonPlusProfile;
  else if (particleType == G4AntiProton::AntiProtonDefinition())
    theProfile = theAntiProtonProfile;
  else if (particleType == G4Proton::ProtonDefinition())
    theProfile = theProtonProfile;

  // input variables for GflashHadronShowerProfile
  G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy() / GeV;
  G4double globalTime = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetGlobalTime();
  G4double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition() / cm;
  G4ThreeVector momentum = fastTrack.GetPrimaryTrack()->GetMomentum() / GeV;
  G4int showerType = Gflash::findShowerType(position);

  theProfile->initialize(showerType, energy, globalTime, charge, position, momentum);
  theProfile->loadParameters();
  theProfile->hadronicParameterization();

  // make hits
  makeHits(fastTrack);
}

void GflashHadronShowerModel::makeHits(const G4FastTrack &fastTrack) {
  std::vector<GflashHit> &gflashHitList = theProfile->getGflashHitList();

  theGflashStep->SetTrack(const_cast<G4Track *>(fastTrack.GetPrimaryTrack()));
  theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(
      const_cast<G4VProcess *>(fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));
  theGflashNavigator->SetWorldVolume(
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

  std::vector<GflashHit>::const_iterator spotIter = gflashHitList.begin();
  std::vector<GflashHit>::const_iterator spotIterEnd = gflashHitList.end();

  for (; spotIter != spotIterEnd; spotIter++) {
    theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(
        spotIter->getPosition(), G4ThreeVector(0, 0, 0), theGflashTouchableHandle, false);
    updateGflashStep(spotIter->getPosition(), spotIter->getTime());

    G4VPhysicalVolume *aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
    if (aCurrentVolume == nullptr)
      continue;

    G4LogicalVolume *lv = aCurrentVolume->GetLogicalVolume();
    if (lv->GetRegion()->GetName() != "CaloRegion")
      continue;

    theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
    G4VSensitiveDetector *aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();

    if (aSensitive == nullptr)
      continue;

    G4String nameCalor = aCurrentVolume->GetName();
    nameCalor.assign(nameCalor, 0, 2);
    G4double samplingWeight = 1.0;
    if (nameCalor == "HB") {
      samplingWeight = Gflash::scaleSensitiveHB;
    } else if (nameCalor == "HE" || nameCalor == "HT") {
      samplingWeight = Gflash::scaleSensitiveHE;
    }
    theGflashStep->SetTotalEnergyDeposit(spotIter->getEnergy() * samplingWeight);

    aSensitive->Hit(theGflashStep);
  }
}

void GflashHadronShowerModel::updateGflashStep(const G4ThreeVector &spotPosition, G4double timeGlobal) {
  theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
  theGflashStep->GetPreStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPostStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
}

G4bool GflashHadronShowerModel::isFirstInelasticInteraction(const G4FastTrack &fastTrack) {
  G4bool isFirst = false;

  G4StepPoint *preStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint();
  G4StepPoint *postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();

  G4String procName = postStep->GetProcessDefinedStep()->GetProcessName();
  G4ParticleDefinition *particleType = fastTrack.GetPrimaryTrack()->GetDefinition();

  //@@@ this part is still temporary and the cut for the variable ratio should
  // be optimized later

  if ((particleType == G4PionPlus::PionPlusDefinition() && procName == "WrappedPionPlusInelastic") ||
      (particleType == G4PionMinus::PionMinusDefinition() && procName == "WrappedPionMinusInelastic") ||
      (particleType == G4KaonPlus::KaonPlusDefinition() && procName == "WrappedKaonPlusInelastic") ||
      (particleType == G4KaonMinus::KaonMinusDefinition() && procName == "WrappedKaonMinusInelastic") ||
      (particleType == G4AntiProton::AntiProtonDefinition() && procName == "WrappedAntiProtonInelastic") ||
      (particleType == G4Proton::ProtonDefinition() && procName == "WrappedProtonInelastic")) {
    // skip to the second interaction if the first inelastic is a quasi-elastic
    // like interaction
    //@@@ the cut may be optimized later

    const G4TrackVector *fSecondaryVector = fastTrack.GetPrimaryTrack()->GetStep()->GetSecondary();
    G4double leadingEnergy = 0.0;

    // loop over 'all' secondaries including those produced by continuous
    // processes.
    //@@@may require an additional condition only for hadron interaction with
    // the process name, but it will not change the result anyway

    for (unsigned int isec = 0; isec < fSecondaryVector->size(); isec++) {
      G4Track *fSecondaryTrack = (*fSecondaryVector)[isec];
      G4double secondaryEnergy = fSecondaryTrack->GetKineticEnergy();

      if (secondaryEnergy > leadingEnergy) {
        leadingEnergy = secondaryEnergy;
      }
    }

    if ((preStep->GetTotalEnergy() != 0) && (leadingEnergy / preStep->GetTotalEnergy() < Gflash::QuasiElasticLike))
      isFirst = true;

    // Fill debugging histograms and check information on secondaries -
    // remove after final implimentation

    if (theHisto->getStoreFlag()) {
      theHisto->preStepPosition->Fill(preStep->GetPosition().getRho() / cm);
      theHisto->postStepPosition->Fill(postStep->GetPosition().getRho() / cm);
      theHisto->deltaStep->Fill((postStep->GetPosition() - preStep->GetPosition()).getRho() / cm);
      theHisto->kineticEnergy->Fill(fastTrack.GetPrimaryTrack()->GetKineticEnergy() / GeV);
      theHisto->energyLoss->Fill(fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy() / GeV));
      theHisto->energyRatio->Fill(leadingEnergy / preStep->GetTotalEnergy());
    }
  }
  return isFirst;
}

G4bool GflashHadronShowerModel::excludeDetectorRegion(const G4FastTrack &fastTrack) {
  G4bool isExcluded = false;
  int verbosity = theParSet.getUntrackedParameter<int>("Verbosity");

  // exclude regions where geometry are complicated
  //+- one supermodule around the EB/EE boundary: 1.479 +- 0.0174*5
  G4double eta = fastTrack.GetPrimaryTrack()->GetPosition().pseudoRapidity();
  if (std::fabs(eta) > 1.392 && std::fabs(eta) < 1.566) {
    if (verbosity > 0) {
      edm::LogInfo("SimGeneralGFlash") << "GflashHadronShowerModel: excluding region of eta = " << eta;
    }
    return true;
  } else {
    G4StepPoint *postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();

    Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(postStep->GetPosition() / cm);
    G4double distOut = 9999.0;

    // exclude the region where the shower starting point is inside the
    // preshower
    if (std::fabs(eta) > Gflash::EtaMin[Gflash::kENCA] &&
        std::fabs((postStep->GetPosition()).getZ() / cm) < Gflash::Zmin[Gflash::kENCA]) {
      return true;
    }

    //<---the shower starting point is always inside envelopes
    //@@@exclude the region where the shower starting point is too close to the
    // end of the hadronic envelopes (may need to be optimized further!)
    //@@@if we extend parameterization including Magnet/HO, we need to change
    // this strategy
    if (kCalor == Gflash::kHB) {
      distOut = Gflash::Rmax[Gflash::kHB] - postStep->GetPosition().getRho() / cm;
      if (distOut < Gflash::MinDistanceToOut)
        isExcluded = true;
    } else if (kCalor == Gflash::kHE) {
      distOut = Gflash::Zmax[Gflash::kHE] - std::fabs(postStep->GetPosition().getZ() / cm);
      if (distOut < Gflash::MinDistanceToOut)
        isExcluded = true;
    }

    //@@@remove this print statement later
    if (isExcluded && verbosity > 0) {
      std::cout << "GflashHadronShowerModel: skipping kCalor = " << kCalor << " DistanceToOut " << distOut << " from ("
                << (postStep->GetPosition()).getRho() / cm << ":" << (postStep->GetPosition()).getZ() / cm
                << ") of KE = " << fastTrack.GetPrimaryTrack()->GetKineticEnergy() / GeV << std::endl;
    }
  }

  return isExcluded;
}

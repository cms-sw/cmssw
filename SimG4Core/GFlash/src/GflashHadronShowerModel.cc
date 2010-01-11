#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashPiKShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashProtonShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashAntiProtonShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashNameSpace.h"
#include "SimGeneral/GFlash/interface/GflashHistogram.h"
#include "SimGeneral/GFlash/interface/GflashHit.h"

#include "G4FastSimulationManager.hh"
#include "G4TransportationManager.hh"
#include "G4TouchableHandle.hh"
#include "G4VSensitiveDetector.hh"
#include "G4VPhysicalVolume.hh"

#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4VProcess.hh"

#include <vector>

GflashHadronShowerModel::GflashHadronShowerModel(G4String modelName, G4Region* envelope, edm::ParameterSet parSet)
  : G4VFastSimulationModel(modelName, envelope), theParSet(parSet)
{
  theProfile = new GflashHadronShowerProfile(parSet);
  thePiKProfile = new GflashPiKShowerProfile(parSet);
  theProtonProfile = new GflashProtonShowerProfile(parSet);
  theAntiProtonProfile = new GflashAntiProtonShowerProfile(parSet);
  theHisto = GflashHistogram::instance();

  theGflashStep = new G4Step();
  theGflashTouchableHandle = new G4TouchableHistory();
  theGflashNavigator = new G4Navigator();

}

GflashHadronShowerModel::~GflashHadronShowerModel()
{
  if(theProfile) delete theProfile;
  if(theGflashStep) delete theGflashStep;
}

G4bool GflashHadronShowerModel::IsApplicable(const G4ParticleDefinition& particleType)
{
  return 
    &particleType == G4PionMinus::PionMinusDefinition() ||
    &particleType == G4PionPlus::PionPlusDefinition() ||
    &particleType == G4KaonMinus::KaonMinusDefinition() ||
    &particleType == G4KaonPlus::KaonPlusDefinition() ||
    &particleType == G4AntiProton::AntiProtonDefinition() ||
    &particleType == G4Proton::ProtonDefinition() ;
}

G4bool GflashHadronShowerModel::ModelTrigger(const G4FastTrack& fastTrack)
{
  // ModelTrigger returns false for Gflash Hadronic Shower Model if it is not
  // tested from the corresponding wrapper process, GflashHadronWrapperProcess. 
  // Temporarily the track status is set to fPostponeToNextEvent at the wrapper
  // process before ModelTrigger is really tested for the second time through 
  // PostStepGPIL of enviaged parameterization processes.  The better 
  // implmentation may be using via G4VUserTrackInformation of each track, which
  // requires to modify a geant source code of stepping (G4SteppingManager2)

  G4bool trigger = false;

  // mininum energy cutoff to parameterize
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() < Gflash::energyCutOff) return trigger;

  // check whether this is called from the normal GPIL or the wrapper process GPIL
  if(fastTrack.GetPrimaryTrack()->GetTrackStatus() == fPostponeToNextEvent ) {

    // Shower pameterization start at the first inelastic interaction point
    G4bool isInelastic  = isFirstInelasticInteraction(fastTrack);
    
    // Other conditions
    if(isInelastic) {
      trigger = (!excludeDetectorRegion(fastTrack));
    }
  }

  return trigger;

}

void GflashHadronShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep)
{
  // kill the particle
  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);
  fastStep.ProposeTotalEnergyDeposited(fastTrack.GetPrimaryTrack()->GetKineticEnergy());

  // parameterize energy depostion by the particle type
  G4ParticleDefinition* particleType = fastTrack.GetPrimaryTrack()->GetDefinition();
  
  theProfile = thePiKProfile;
  if(particleType == G4AntiProton::AntiProtonDefinition()) theProfile = theAntiProtonProfile;
  else if(particleType == G4Proton::ProtonDefinition()) theProfile = theProtonProfile;

  //input variables for GflashHadronShowerProfile
  G4int showerType = findShowerType(fastTrack);
  G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4double globalTime = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetGlobalTime();
  G4double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  G4ThreeVector momentum = fastTrack.GetPrimaryTrack()->GetMomentum()/GeV;

  theProfile->initialize(showerType,energy,globalTime,charge,position,momentum);
  theProfile->loadParameters();
  theProfile->hadronicParameterization();

  // make hits
  makeHits(fastTrack);

}

void GflashHadronShowerModel::makeHits(const G4FastTrack& fastTrack) {

  std::vector<GflashHit>& gflashHitList = theProfile->getGflashHitList();

  theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));
  theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*>
    (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));
  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

  std::vector<GflashHit>::const_iterator spotIter    = gflashHitList.begin();
  std::vector<GflashHit>::const_iterator spotIterEnd = gflashHitList.end();

  for( ; spotIter != spotIterEnd; spotIter++){

    theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(spotIter->getPosition(),G4ThreeVector(0,0,0),
								  theGflashTouchableHandle, false);
    updateGflashStep(spotIter->getPosition(),spotIter->getTime());

    G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
    if( aCurrentVolume == 0 ) continue;

    G4LogicalVolume* lv = aCurrentVolume->GetLogicalVolume();
    if(lv->GetRegion()->GetName() != "CaloRegion") continue;
	  
    theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
    G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();

    if( aSensitive == 0 ) continue;
    
    G4String nameCalor = aCurrentVolume->GetName();
    nameCalor.assign(nameCalor,0,2);
    G4double samplingWeight = 1.0; 
    if(nameCalor == "HB" || nameCalor=="HE" || nameCalor == "HT") samplingWeight = Gflash::scaleSensitive;
    
    theGflashStep->SetTotalEnergyDeposit(spotIter->getEnergy()*samplingWeight);

    aSensitive->Hit(theGflashStep);

  }
}

void GflashHadronShowerModel::updateGflashStep(G4ThreeVector spotPosition, G4double timeGlobal)
{
  theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
  theGflashStep->GetPreStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPostStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
}

G4bool GflashHadronShowerModel::isFirstInelasticInteraction(const G4FastTrack& fastTrack)
{
  G4bool isFirst=false;

  G4StepPoint* preStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint();
  G4StepPoint* postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();

  G4String procName = postStep->GetProcessDefinedStep()->GetProcessName();  
  G4ParticleDefinition* particleType = fastTrack.GetPrimaryTrack()->GetDefinition();

  //@@@ this part is still temporary and the cut for the variable ratio should be optimized later

  if((particleType == G4PionPlus::PionPlusDefinition() && procName == "WrappedPionPlusInelastic") || 
     (particleType == G4PionMinus::PionMinusDefinition() && procName == "WrappedPionMinusInelastic") ||
     (particleType == G4KaonPlus::KaonPlusDefinition() && procName == "WrappedKaonPlusInelastic") ||
     (particleType == G4KaonMinus::KaonMinusDefinition() && procName == "WrappedKaonMinusInelastic") ||
     (particleType == G4AntiProton::AntiProtonDefinition() && procName == "WrappedAntiProtonInelastic") ||
     (particleType == G4Proton::ProtonDefinition() && procName == "WrappedProtonInelastic")) {

    //skip to the second interaction if the first inelastic is a quasi-elastic like interaction
    //@@@ the cut may be optimized later

    const G4TrackVector* fSecondaryVector = fastTrack.GetPrimaryTrack()->GetStep()->GetSecondary();
    G4double leadingEnergy = 0.0;

    //loop over 'all' secondaries including those produced by continuous processes.
    //@@@may require an additional condition only for hadron interaction with the process name,
    //but it will not change the result anyway

    for (unsigned int isec = 0 ; isec < fSecondaryVector->size() ; isec++) {
      G4Track* fSecondaryTrack = (*fSecondaryVector)[isec];
      G4double secondaryEnergy = fSecondaryTrack->GetKineticEnergy();

      if(secondaryEnergy > leadingEnergy ) {
        leadingEnergy = secondaryEnergy;
      }
    }

    if((preStep->GetTotalEnergy()!=0) && 
       (leadingEnergy/preStep->GetTotalEnergy() < Gflash::QuasiElasticLike)) isFirst = true;

    //Fill debugging histograms and check information on secondaries -
    //remove after final implimentation

    if(theHisto->getStoreFlag()) {
      theHisto->preStepPosition->Fill(preStep->GetPosition().getRho()/cm);
      theHisto->postStepPosition->Fill(postStep->GetPosition().getRho()/cm);
      theHisto->deltaStep->Fill((postStep->GetPosition() - preStep->GetPosition()).getRho()/cm);
      theHisto->kineticEnergy->Fill(fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV);
      theHisto->energyLoss->Fill(fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy()/GeV));
      theHisto->energyRatio->Fill(leadingEnergy/preStep->GetTotalEnergy());
    }

 }
  return isFirst;
} 

G4bool GflashHadronShowerModel::excludeDetectorRegion(const G4FastTrack& fastTrack)
{
  G4bool isExcluded=false;
  int verbosity = theParSet.getUntrackedParameter<int>("Verbosity");
  
  //exclude regions where geometry are complicated 
  G4double eta =   fastTrack.GetPrimaryTrack()->GetPosition().pseudoRapidity() ;
  if(fabs(eta) > Gflash::EtaMax[Gflash::kESPM] && fabs(eta) < Gflash::EtaMin[Gflash::kENCA]) {
    if(verbosity>0) {
       edm::LogInfo("SimGeneralGFlash") << "GflashHadronShowerModel: excluding region of eta = " << eta;
    }
    return true;  
  }
  else {
    G4StepPoint* postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();

    Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(postStep->GetPosition()/cm);
    G4double distOut = 9999.0;
    //exclude the region where the shower starting point is outside parameterization envelopes
    if(kCalor==Gflash::kNULL) {
      isExcluded = true;
    }
    //@@@exclude the region where the shower starting point is too close to the end of
    //the hadronic envelopes (may need to be optimized further!)
    //@@@if we extend parameterization including Magnet/HO, we need to change this strategy
    else if(kCalor == Gflash::kHB) {
      distOut =  Gflash::Rmax[Gflash::kHB] - postStep->GetPosition().getRho()/cm;
      if (distOut < Gflash::MinDistanceToOut ) isExcluded = true;
    }
    else if(kCalor == Gflash::kHE) {
      distOut =  Gflash::Zmax[Gflash::kHE] - std::fabs(postStep->GetPosition().getZ()/cm);
      if (distOut < Gflash::MinDistanceToOut ) isExcluded = true;
    }

    //@@@remove this print statement later
    if(isExcluded && verbosity > 0) {
      std::cout << "GflashHadronShowerModel: skipping kCalor = " << kCalor << 
	" DistanceToOut " << distOut << " from (" <<  (postStep->GetPosition()).getRho()/cm << 
	":" << (postStep->GetPosition()).getZ()/cm << ") of KE = " << fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV << std::endl;
    }
  }

  return isExcluded;
}

G4int GflashHadronShowerModel::findShowerType(const G4FastTrack& fastTrack)
{
  // Initialization of longitudinal and lateral parameters for 
  // hadronic showers. Simulation of the intrinsic fluctuations

  // type of hadron showers subject to the shower starting point (ssp)
  // showerType = -1 : default (invalid) 
  // showerType =  0 : ssp before EBRY (barrel crystal) 
  // showerType =  1 : ssp inside EBRY
  // showerType =  2 : ssp after  EBRY before HB
  // showerType =  3 : ssp inside HB
  // showerType =  4 : ssp before EFRY (endcap crystal) 
  // showerType =  5 : ssp inside EFRY 
  // showerType =  6 : ssp after  EFRY before HE
  // showerType =  7 : ssp inside HE
    
  G4TouchableHistory* touch = (G4TouchableHistory*)(fastTrack.GetPrimaryTrack()->GetTouchable());
  G4LogicalVolume* lv = touch->GetVolume()->GetLogicalVolume();

  std::size_t pos1  = lv->GetName().find("EBRY");
  std::size_t pos11 = lv->GetName().find("EWAL");
  std::size_t pos12 = lv->GetName().find("EWRA");
  std::size_t pos2  = lv->GetName().find("EFRY");

  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(position);

  G4int showerType = -1;

  //central
  if (kCalor == Gflash::kESPM || kCalor == Gflash::kHB ) {

    G4double posRho = position.getRho();

    if(pos1 != std::string::npos || pos11 != std::string::npos || pos12 != std::string::npos ) {
      showerType = 1;
    }
    else {
      if(kCalor == Gflash::kESPM) {
	showerType = 2;
	if( posRho < Gflash::Rmin[Gflash::kESPM]+ Gflash::ROffCrystalEB ) showerType = 0;
      }
      else showerType = 3;
    }

  }
  //forward
  else if (kCalor == Gflash::kENCA || kCalor == Gflash::kHE) {
    if(pos2 != std::string::npos) {
      showerType = 5;
    }
    else {
      if(kCalor == Gflash::kENCA) {
	showerType = 6;
	if(fabs(position.getZ()) < Gflash::Zmin[Gflash::kENCA] + Gflash::ZOffCrystalEE) showerType = 4;
      }
      else showerType = 7;
    }
    //@@@need z-dependent correction on the mean energy reponse
  }

  return showerType;
}

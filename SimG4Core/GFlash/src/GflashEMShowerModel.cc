//
// initial setup : E.Barberio & Joanna Weng
// big changes : Soon Jun & Dongwook Jang
//
#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"

#include "SimGeneral/GFlash/interface/GflashEMShowerProfile.h"
#include "SimGeneral/GFlash/interface/GflashHit.h"

#include "G4Electron.hh"
#include "G4EventManager.hh"
#include "G4FastSimulationManager.hh"
#include "G4LogicalVolume.hh"
#include "G4Positron.hh"
#include "G4TouchableHandle.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"
#include "G4VSensitiveDetector.hh"

using namespace CLHEP;

GflashEMShowerModel::GflashEMShowerModel(const G4String &modelName,
                                         G4Envelope *envelope,
                                         const edm::ParameterSet &parSet)
    : G4VFastSimulationModel(modelName, envelope), theParSet(parSet) {
  theProfile = new GflashEMShowerProfile(parSet);
  theRegion = reinterpret_cast<G4Region *>(envelope);

  theGflashStep = new G4Step();
  theGflashTouchableHandle = new G4TouchableHistory();
  theGflashNavigator = new G4Navigator();
}

// -----------------------------------------------------------------------------------

GflashEMShowerModel::~GflashEMShowerModel() {
  delete theProfile;
  delete theGflashStep;
}

G4bool GflashEMShowerModel::IsApplicable(const G4ParticleDefinition &particleType) {
  return (&particleType == G4Electron::ElectronDefinition() || &particleType == G4Positron::PositronDefinition());
}

// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::ModelTrigger(const G4FastTrack &fastTrack) {
  // Mininum energy cutoff to parameterize
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() < GeV) {
    return false;
  }
  if (excludeDetectorRegion(fastTrack)) {
    return false;
  }

  // This will be changed accordingly when the way
  // dealing with CaloRegion changes later.
  const G4VPhysicalVolume *pCurrentVolume = (fastTrack.GetPrimaryTrack()->GetTouchable())->GetVolume();
  if (pCurrentVolume == nullptr) {
    return false;
  }

  const G4LogicalVolume *lv = pCurrentVolume->GetLogicalVolume();
  if (lv->GetRegion() != theRegion) {
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------------
void GflashEMShowerModel::DoIt(const G4FastTrack &fastTrack, G4FastStep &fastStep) {
  // Kill the parameterised particle:
  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);

  // input variables for GflashEMShowerProfile with showerType = 1,5 (shower
  // starts inside crystals)
  G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy() / GeV;
  G4double globalTime = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetGlobalTime();
  G4double charge = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetCharge();
  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition() / cm;
  G4ThreeVector momentum = fastTrack.GetPrimaryTrack()->GetMomentum() / GeV;
  G4int showerType = Gflash::findShowerType(position);

  // Do actual parameterization. The result of parameterization is gflashHitList
  theProfile->initialize(showerType, energy, globalTime, charge, position, momentum);
  theProfile->parameterization();

  // make hits
  makeHits(fastTrack);
}

void GflashEMShowerModel::makeHits(const G4FastTrack &fastTrack) {
  std::vector<GflashHit> &gflashHitList = theProfile->getGflashHitList();

  theGflashStep->SetTrack(const_cast<G4Track *>(fastTrack.GetPrimaryTrack()));

  theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(
      const_cast<G4VProcess *>(fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));
  theGflashNavigator->SetWorldVolume(
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

  std::vector<GflashHit>::const_iterator spotIter = gflashHitList.begin();
  std::vector<GflashHit>::const_iterator spotIterEnd = gflashHitList.end();

  for (; spotIter != spotIterEnd; spotIter++) {
    // put touchable for each hit so that touchable history keeps track of each
    // step.
    theGflashNavigator->LocateGlobalPointAndUpdateTouchableHandle(
        spotIter->getPosition(), G4ThreeVector(0, 0, 0), theGflashTouchableHandle, false);
    updateGflashStep(spotIter->getPosition(), spotIter->getTime());

    // Send G4Step information to Hit/Digi if the volume is sensitive
    // Copied from G4SteppingManager.cc

    G4VPhysicalVolume *aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();
    if (aCurrentVolume == nullptr)
      continue;

    G4LogicalVolume *lv = aCurrentVolume->GetLogicalVolume();
    if (lv->GetRegion() != theRegion)
      continue;

    theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
    G4VSensitiveDetector *aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();

    if (aSensitive == nullptr)
      continue;

    theGflashStep->SetTotalEnergyDeposit(spotIter->getEnergy());
    aSensitive->Hit(theGflashStep);
  }
}

void GflashEMShowerModel::updateGflashStep(const G4ThreeVector &spotPosition, G4double timeGlobal) {
  theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
  theGflashStep->GetPreStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPostStepPoint()->SetPosition(spotPosition);
  theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
}

// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::excludeDetectorRegion(const G4FastTrack &fastTrack) {
  // exclude regions where geometry are complicated
  //+- one supermodule around the EB/EE boundary: 1.479 +- 0.0174*5
  G4double eta = fastTrack.GetPrimaryTrack()->GetPosition().pseudoRapidity();
  return (std::fabs(eta) > 1.392 && std::fabs(eta) < 1.566);
}

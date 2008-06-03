//
// initial setup : E.Barberio & Joanna Weng 
// big changes : Soon Jun & Dongwook Jang

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4VProcess.hh"
#include "G4VPhysicalVolume.hh" 
#include "G4LogicalVolume.hh"
#include "G4VSensitiveDetector.hh"
#include "G4TransportationManager.hh"
#include "G4RegionStore.hh"
#include "G4FastSimulationManager.hh"

#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashEMShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"

GflashEMShowerModel::GflashEMShowerModel(G4String modelName, G4Envelope* envelope, edm::ParameterSet parSet)
  : G4VFastSimulationModel(modelName, envelope), theParSet(parSet) {

  theProfile = new GflashEMShowerProfile(envelope,parSet);
  theGflashStep = new G4Step();
  theGflashNavigator = new G4Navigator();
  theGflashTouchableHandle = new G4TouchableHistory();

  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

}

// -----------------------------------------------------------------------------------

GflashEMShowerModel::~GflashEMShowerModel() {

  if(theProfile) delete theProfile;
  if(theGflashStep) delete theGflashStep;
  if(theGflashNavigator) delete theGflashNavigator;
}

G4bool GflashEMShowerModel::IsApplicable(const G4ParticleDefinition& particleType) { 

  return ( &particleType == G4Electron::ElectronDefinition() ||
	   &particleType == G4Positron::PositronDefinition() ); 

}

// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::ModelTrigger(const G4FastTrack & fastTrack ) {

  // Mininum energy cutoff to parameterize
  if(fastTrack.GetPrimaryTrack()->GetKineticEnergy() < 1.0*GeV) return false;
  if(excludeDetectorRegion(fastTrack)) return false;

  G4bool trigger = fastTrack.GetPrimaryTrack()->GetDefinition() == G4Electron::ElectronDefinition() || 
    fastTrack.GetPrimaryTrack()->GetDefinition() == G4Positron::PositronDefinition();

  if(!trigger) return false;

  // This will be changed accordingly when the way dealing with GflashRegion changes later.
  G4TouchableHistory* touch = (G4TouchableHistory*)(fastTrack.GetPrimaryTrack()->GetTouchable());
  G4VPhysicalVolume* pCurrentVolume = touch->GetVolume();
  if( pCurrentVolume == 0) return false;

  G4LogicalVolume* lv = pCurrentVolume->GetLogicalVolume();
  if(lv->GetRegion()->GetName() != "GflashRegion") return false;

  // The parameterization starts inside crystals
  std::size_t pos1 = lv->GetName().find("EBRY");
  std::size_t pos2 = lv->GetName().find("EFRY");
  if(pos1 == std::string::npos && pos2 == std::string::npos) return false;

  return true;

}


// -----------------------------------------------------------------------------------
void GflashEMShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) {

  // Initialize energySpotList
  theProfile->clearSpotList();

  // Do actual parameterization. The result of parameterization is energySpotList
  theProfile->parameterization(fastTrack);
  std::vector<GflashEnergySpot>& energySpotList = theProfile->getEnergySpotList();

  // The following procedure is creating fake G4Steps from GflashEnergySpot.
  // The time is not meaningful but G4Step requires that information to make a step unique.
  // Uniqueness of G4Step is important otherwise hits won't be created.
  G4double timeGlobal = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetGlobalTime();

  std::vector<GflashEnergySpot>::const_iterator spotIter    = energySpotList.begin();
  std::vector<GflashEnergySpot>::const_iterator spotIterEnd = energySpotList.end();

  for( ; spotIter != spotIterEnd; spotIter++){

    // to make a different time for each fake step. (0.03 nsec is corresponding to 1cm step size)
    timeGlobal += 0.0001*nanosecond;

    // fill equivalent changes to a (fake) step associated with a spot 

    theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));
    theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
    theGflashStep->GetPreStepPoint()->SetPosition(spotIter->getPosition());
    theGflashStep->GetPostStepPoint()->SetPosition(spotIter->getPosition());
    theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));

    //put touchable for each energy spot so that touchable history keeps track of each step.
    theGflashNavigator->LocateGlobalPointAndUpdateTouchable(spotIter->getPosition(),theGflashTouchableHandle(), false);
    theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
    theGflashStep->SetTotalEnergyDeposit(spotIter->getEnergy());
    
    // Send G4Step information to Hit/Digi if the volume is sensitive
    // Copied from G4SteppingManager.cc
    
    G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();

    if( aCurrentVolume != 0 ) {
      theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
      G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();
      
      if( aSensitive != 0 ) {
	aSensitive->Hit(theGflashStep);
      }
    }
  }

  // Kill the parameterised particle:
  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);

}


// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::excludeDetectorRegion(const G4FastTrack& fastTrack) {

  G4bool isExcluded=false;

  //exclude regions where geometry are complicated
  G4double eta =   fastTrack.GetPrimaryTrack()->GetMomentum().pseudoRapidity() ;
  if(fabs(eta) > 1.3 && fabs(eta) < 1.57) return true;

  return isExcluded;
}

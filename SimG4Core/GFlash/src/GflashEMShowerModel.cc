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

GflashEMShowerModel::GflashEMShowerModel(G4String modelName, G4Envelope* envelope)
  : G4VFastSimulationModel(modelName, envelope) {

  theProfile = new GflashEMShowerProfile(envelope);
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

  // Here we switch region from the default to GflashRegion and associate it to G4FastSimulationManger.
  // The perfect place to do this is in CaloModel.cc but GflashRegion is not defined there.
  // So this is a temporary solution. We will make an elegant way later...

  static G4Region* gflashRegion = 0;
  if(gflashRegion == 0){
    gflashRegion = G4RegionStore::GetInstance()->GetRegion("GflashRegion");
    gflashRegion->SetFastSimulationManager(G4RegionStore::GetInstance()->GetRegion("DefaultRegionForTheWorld")->GetFastSimulationManager());
  }

  // mininum energy cutoff to parameterize
  if(fastTrack.GetPrimaryTrack()->GetKineticEnergy() < 1.0*GeV) return false;
  if(excludeDetectorRegion(fastTrack)) return false;

  G4bool trigger = fastTrack.GetPrimaryTrack()->GetDefinition() == G4Electron::ElectronDefinition() || 
    fastTrack.GetPrimaryTrack()->GetDefinition() == G4Positron::PositronDefinition();

  if(!trigger) return false;

  // this fixes the width of energy contaiment.
  G4TouchableHistory* touch = (G4TouchableHistory*)(fastTrack.GetPrimaryTrack()->GetTouchable());
  G4VPhysicalVolume* pCurrentVolume = touch->GetVolume();
  if( pCurrentVolume == 0) return false;

  if(pCurrentVolume->GetLogicalVolume()->GetRegion()->GetName() != "GflashRegion") return false;
  if(pCurrentVolume->GetName() != "EBRY" && pCurrentVolume->GetName() != "EFRY") return false;
  //  if(fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() != "eBrem") return false;

  return true;

}


// -----------------------------------------------------------------------------------
void GflashEMShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) {

  // Parameterize shower shape and get resultant energy spots
  theProfile->clearSpotList();
  theProfile->parameterization(fastTrack);

  std::vector<GflashEnergySpot>& energySpotList = theProfile->getEnergySpotList();

  // Make hits
  G4double timeGlobal = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetGlobalTime();

  std::vector<GflashEnergySpot>::const_iterator spotIter    = energySpotList.begin();
  std::vector<GflashEnergySpot>::const_iterator spotIterEnd = energySpotList.end();

  for( ; spotIter != spotIterEnd; spotIter++){

    // to make a different time for each fake step. (+1.0 is arbitrary)
    timeGlobal += 1.0;

    // fill equivalent changes to a (fake) step associated with a spot 

    theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));
    theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
    theGflashStep->GetPreStepPoint()->SetPosition(spotIter->getPosition());
    theGflashStep->GetPostStepPoint()->SetPosition(spotIter->getPosition());
    theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));

    //put touchable for each energy spot
    theGflashNavigator->LocateGlobalPointAndUpdateTouchable(spotIter->getPosition(),theGflashTouchableHandle(), false);
    theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
    theGflashStep->SetTotalEnergyDeposit(spotIter->getEnergy());
    
    // Send G4Step information to Hit/Dig if the volume is sensitive
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

  /*
  //exclude the region where the shower starting point is too close to the end of
  //the hadronic envelopes (may need to be optimized further!)

  GflashCalorimeterNumber kColor = theProfile->getCalorimeterNumber(fastTrack);

  //@@@ need a proper scale
  const G4double minDistantToOut = 10.;

  if(kColor == kESPM || kColor == kENCA) {
    G4double distOut = fastTrack.GetEnvelopeSolid()->
      DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
                    fastTrack.GetPrimaryTrackLocalDirection());
    if (distOut < minDistantToOut ) isExcluded = true;
  }
  */

  return isExcluded;
}

//
// initial setup : E.Barberio & Joanna Weng 
// big changes : Soon Jun & Dongwook Jang

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4VProcess.hh"
#include "G4VPhysicalVolume.hh" 
#include "G4LogicalVolume.hh"
#include "G4TransportationManager.hh"
#include "G4RegionStore.hh"
#include "G4FastSimulationManager.hh"

#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashEMShowerProfile.h"

GflashEMShowerModel::GflashEMShowerModel(G4String modelName, G4Envelope* envelope, edm::ParameterSet parSet)
  : G4VFastSimulationModel(modelName, envelope), theParSet(parSet) {

  theProfile = new GflashEMShowerProfile(envelope,parSet);

}

// -----------------------------------------------------------------------------------

GflashEMShowerModel::~GflashEMShowerModel() {

  if(theProfile) delete theProfile;
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

  // This will be changed accordingly when the way dealing with CaloRegion changes later.
  G4TouchableHistory* touch = (G4TouchableHistory*)(fastTrack.GetPrimaryTrack()->GetTouchable());
  G4VPhysicalVolume* pCurrentVolume = touch->GetVolume();
  if( pCurrentVolume == 0) return false;

  G4LogicalVolume* lv = pCurrentVolume->GetLogicalVolume();
  if(lv->GetRegion()->GetName() != "CaloRegion") return false;

  // The parameterization starts inside crystals
  std::size_t pos1 = lv->GetName().find("EBRY");
  std::size_t pos2 = lv->GetName().find("EFRY");
  /*
  std::size_t pos3 = lv->GetName().find("HVQ");
  std::size_t pos4 = lv->GetName().find("HF");
  if(pos1 == std::string::npos && pos2 == std::string::npos &&
     pos3 == std::string::npos && pos4 == std::string::npos) return false;
  */
  //@@@for now, HF is not a part of Gflash Envelopes
  if(pos1 == std::string::npos && pos2 == std::string::npos ) return false;

  return true;

}


// -----------------------------------------------------------------------------------
void GflashEMShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) {

  // Do actual parameterization. The result of parameterization is energySpotList
  theProfile->parameterization(fastTrack);

  // Kill the parameterised particle:
  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);

}


// -----------------------------------------------------------------------------------
G4bool GflashEMShowerModel::excludeDetectorRegion(const G4FastTrack& fastTrack) {

  G4bool isExcluded=false;

  //exclude regions where geometry are complicated
  G4double eta =   fastTrack.GetPrimaryTrack()->GetPosition().pseudoRapidity() ;
  if(fabs(eta) > 1.3 && fabs(eta) < 1.57) return true;

  return isExcluded;
}

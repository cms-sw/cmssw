#include "SimG4Core/GFlash/interface/GFlash.h"
#include "SimG4Core/GFlash/interface/CaloModel.h"
#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4NeutronTrackingCut.hh"

#include "SimG4Core/GFlash/interface/HadronPhysicsQGSP_WP.h"
#include "G4DataQuestionaire.hh"

GFlash::GFlash(G4LogicalVolumeToDDLogicalPartMap& map,
	       const edm::ParameterSet & p) : PhysicsList(map, p), 
					      caloModel(0) {
  G4DataQuestionaire it(photon);

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP 3.3 + CMS GFLASH\n";

  if (caloModel==0) caloModel = new CaloModel(map, p);

  RegisterPhysics(new ParametrisedPhysics("parametrised")); 

  // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics("standard EM",ver));

  // Synchroton Radiation & GN Physics
  RegisterPhysics(new G4EmExtraPhysics("extra EM"));

  // Decays
  RegisterPhysics(new G4DecayPhysics("decay",ver));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics("elastic",ver,false)); 

  // Hadron Physics
  G4bool quasiElastic=true;
  RegisterPhysics(new HadronPhysicsQGSP_WP("hadron",quasiElastic));
  //RegisterPhysics(new HadronPhysicsQGSP("hadron"));

  // Stopping Physics
  RegisterPhysics(new G4QStoppingPhysics("stopping"));

  // Ion Physics
  RegisterPhysics(new G4IonPhysics("ion"));

  // Neutron tracking cut
  RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));
}

GFlash::~GFlash() { if (caloModel!=0) delete caloModel; }


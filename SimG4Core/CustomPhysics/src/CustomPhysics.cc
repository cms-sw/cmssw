#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics71.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsQGSP.hh"
 
CustomPhysics::CustomPhysics(G4LogicalVolumeToDDLogicalPartMap& map,
			     const edm::ParameterSet & p) : PhysicsList(map,p){

  G4DataQuestionaire it(photon);

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP_EMV 3.3 with Flags for EM Physics "
			      << emPhys << " and for Hadronic Physics "
			      << hadPhys << "\n";

  // EM Physics
  RegisterPhysics(new G4EmStandardPhysics71("standard EM v71",ver));

  // Synchroton Radiation & GN Physics
  RegisterPhysics(new G4EmExtraPhysics("extra EM"));

  // Decays
  RegisterPhysics(new G4DecayPhysics("decay"));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics("elastic",ver,false)); 

  // Hadron Physics
  G4bool quasiElastic=true;
  RegisterPhysics(new HadronPhysicsQGSP("hadron",quasiElastic));
  //RegisterPhysics(new HadronPhysicsQGSP("hadron"));

  // Stopping Physics
  RegisterPhysics(new G4QStoppingPhysics("stopping"));

  // Ion Physics
  RegisterPhysics(new G4IonPhysics("ion"));

  // Neutron tracking cut
  RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));

  // Custom Physics
  RegisterPhysics(new CustomPhysicsList("custom",p));    
}

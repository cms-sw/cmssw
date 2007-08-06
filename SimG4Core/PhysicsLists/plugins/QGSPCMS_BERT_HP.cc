#include "QGSPCMS_BERT_HP.hh"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsQGSP_BERT_HP.hh"

QGSPCMS_BERT_HP::QGSPCMS_BERT_HP(G4LogicalVolumeToDDLogicalPartMap& map,
				 const edm::ParameterSet & p) : PhysicsList(map, p) {

  G4DataQuestionaire it(photon);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: QGSP_BERT_HP 2.3\n";
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new G4EmStandardPhysics("standard EM",ver));

    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics("extra EM"));
  }

  // Decays
  RegisterPhysics( new G4DecayPhysics("decay",ver) );

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics("elastic",ver,true));
    
    // Hadron Physics
    //G4bool quasiElastic=true;
    //RegisterPhysics( new HadronPhysicsQGSP_BERT_HP("hadron",quasiElastic));
    RegisterPhysics( new HadronPhysicsQGSP_BERT_HP("hadron"));

    // Stopping Physics
    RegisterPhysics( new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));
  }
}


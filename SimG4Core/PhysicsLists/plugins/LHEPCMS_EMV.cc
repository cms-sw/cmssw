#include "LHEPCMS_EMV.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics71.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsLHEP_EMV.hh"

LHEPCMS_EMV::LHEPCMS_EMV(G4LogicalVolumeToDDLogicalPartMap& map, 
			 const edm::ParameterSet & p) : PhysicsList(map, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "LHEP 4.2 with Flags for EM Physics "
			      << emPhys << " and for Hadronic Physics "
			      << hadPhys << "\n";

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics71("standard EM v71",ver));

    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics("extra EM"));
  }

  // General Physics - i.e. decay
  RegisterPhysics( new G4DecayPhysics("decay"));

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics("LElastic",ver,false));

    // Hadron Physics
    RegisterPhysics(  new HadronPhysicsLHEP_EMV("hadron"));

    // Stopping Physics
    RegisterPhysics(new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));
  }
}


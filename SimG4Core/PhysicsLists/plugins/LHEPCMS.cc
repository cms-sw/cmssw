#include "LHEPCMS.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4HadronElasticPhysics.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsLHEP.hh"

LHEPCMS::LHEPCMS(G4LogicalVolumeToDDLogicalPartMap& map,
		 const HepPDT::ParticleDataTable * table_, 
		 sim::FieldBuilder *fieldBuilder_, 
		 const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  double charge= p.getUntrackedParameter<double>("MonopoleCharge",1.0);
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "LHEP 4.2 with Flags for EM Physics "
			      << emPhys << " and for Hadronic Physics "
			      << hadPhys << "\n";

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics("standard EM",ver));

    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics("extra EM"));
  }

  // General Physics - i.e. decay
  RegisterPhysics( new G4DecayPhysics("decay"));

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics("LElastic",ver,false));

    // Hadron Physics
    RegisterPhysics(  new HadronPhysicsLHEP("hadron"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,fieldBuilder_,charge,ver));
}



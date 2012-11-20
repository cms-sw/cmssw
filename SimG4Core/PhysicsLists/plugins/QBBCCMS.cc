#include "QBBCCMS.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4QStoppingPhysics.hh"
//#include "G4LHEPStoppingPhysics.hh" 

#include "G4DataQuestionaire.hh"
#include "G4HadronInelasticQBBC.hh"
#include "G4HadronElasticPhysics.hh"
//#include "G4HadronDElasticPhysics.hh"
//#include "G4HadronHElasticPhysics.hh"
#include "G4IonBinaryCascadePhysics.hh"
//#include "G4IonPhysics.hh"
#include "G4NeutronTrackingCut.hh"

QBBCCMS::QBBCCMS(G4LogicalVolumeToDDLogicalPartMap& map, 
		 const HepPDT::ParticleDataTable * table_, 
		 sim::FieldBuilder *fieldBuilder_, 
		 const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool ftf     = p.getUntrackedParameter<bool>("FlagFTF",false);
  bool bert    = p.getUntrackedParameter<bool>("FlagBERT",false);
  bool chips   = p.getUntrackedParameter<bool>("FlagCHIPS",false);
  bool hp      = p.getUntrackedParameter<bool>("FlagHP",false);
  bool glauber = p.getUntrackedParameter<bool>("FlagGlauber",false);
  bool tracking= p.getParameter<bool>("TrackingCut");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QBBC 3.1 with Flags for EM Physics "
			      << emPhys << " and for Hadronic Physics "
			      << hadPhys << " Flags for FTF " << ftf
			      << " BERT " << bert << " CHIPS " << chips
			      << " HP " << hp << " Glauber " << glauber
			      << " and tracking cut " << tracking;

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics("standard EM",ver));

    // Synchroton Radiation & GN Physics
    RegisterPhysics(new G4EmExtraPhysics("extra EM"));
  }

  // Decays
  RegisterPhysics(new G4DecayPhysics("decay",ver));

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics("hElastic",ver,false,true));

    // Hadron Physics
    RegisterPhysics( new G4HadronInelasticQBBC("inelastic", ver, ftf,
					       bert, chips, hp, glauber));

    // Stopping Physics
    RegisterPhysics(new G4QStoppingPhysics("stopping",ver));

    // Ion Physics
    RegisterPhysics(new G4IonBinaryCascadePhysics("ionBIC"));

    // Neutron tracking cut
    if (tracking) 
      RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,fieldBuilder_,p));
}


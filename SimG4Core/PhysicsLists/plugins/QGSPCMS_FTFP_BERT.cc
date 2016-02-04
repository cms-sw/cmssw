#include "QGSPCMS_FTFP_BERT.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
//#include "HadronPhysicsQGSP_FTFP_BERT.hh"
#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSPCMS_FTFP_BERT.h"

QGSPCMS_FTFP_BERT::QGSPCMS_FTFP_BERT(G4LogicalVolumeToDDLogicalPartMap& map,
				     const HepPDT::ParticleDataTable * table_,
				     sim::FieldBuilder *fieldBuilder_, 
				     const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP_FTFP_BERT 3.3 with Flags for EM Physics "
			      << emPhys << ", for Hadronic Physics "
			      << hadPhys << " and tracking cut " << tracking;

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics("standard EM",ver));

    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics("extra EM"));
  }

  // Decays
  RegisterPhysics( new G4DecayPhysics("decay",ver) );

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics("elastic",ver,false));

    // Hadron Physics
    G4bool quasiElastic=true;
    RegisterPhysics( new HadronPhysicsQGSPCMS_FTFP_BERT("hadron",quasiElastic)); 
    // RegisterPhysics( new HadronPhysicsQGSP_FTFP_BERT("hadron",quasiElastic));
  
    // Stopping Physics
    RegisterPhysics( new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));

    // Neutron tracking cut
    if (tracking) 
      RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,fieldBuilder_,p));
}

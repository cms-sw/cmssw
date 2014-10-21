#include "QGSPCMS_FTFP_BERT_EML_New.hh"
//#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics95msc93.h"
//#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsLPM.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4EmStandardPhysics_option1.hh"
#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

#include "G4DataQuestionaire.hh"
#include "G4HadronPhysicsQGSP_FTFP_BERT.hh"

QGSPCMS_FTFP_BERT_EML_New::QGSPCMS_FTFP_BERT_EML_New(G4LogicalVolumeToDDLogicalPartMap& map, 
			   const HepPDT::ParticleDataTable * table_,
			   sim::FieldBuilder *fieldBuilder_, 
			   const edm::ParameterSet & p) 
  : PhysicsList(map, table_, fieldBuilder_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  bool munucl  = p.getParameter<bool>("FlagMuNucl");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP_FTFP_BERT_EML_New with Flags for EM Physics "
			      << emPhys << ", for Hadronic Physics "
			      << hadPhys << " and tracking cut " << tracking;

  if (emPhys) {
    // EM Physics
    //RegisterPhysics(new CMSEmStandardPhysics95msc93("EM standard msc93",ver,""));
    RegisterPhysics( new G4EmStandardPhysics_option1(ver));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    if(munucl) { G4String yes = "on"; gn->MuonNuclear(yes); }
    RegisterPhysics(gn);
  }

  // Decays
  this->RegisterPhysics( new G4DecayPhysics(ver));

  if (hadPhys) {
    G4HadronicProcessStore::Instance()->SetVerbose(ver);

    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics(ver));

    // Hadron Physics
    RegisterPhysics(  new G4HadronPhysicsQGSP_FTFP_BERT(ver));

    // Stopping Physics
    RegisterPhysics( new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics(ver));

    // Neutron tracking cut
    if (tracking) {
      RegisterPhysics( new G4NeutronTrackingCut(ver));
    }
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,fieldBuilder_,p));
}


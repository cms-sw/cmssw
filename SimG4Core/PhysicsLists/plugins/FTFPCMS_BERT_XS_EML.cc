#include "FTFPCMS_BERT_XS_EML.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsXS.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysicsXS.hh"
#include "G4NeutronCrossSectionXS.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

#include "G4DataQuestionaire.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"

FTFPCMS_BERT_XS_EML::FTFPCMS_BERT_XS_EML(G4LogicalVolumeToDDLogicalPartMap& map, 
			   const HepPDT::ParticleDataTable * table_,
			   sim::ChordFinderSetter *chordFinderSetter_, 
			   const edm::ParameterSet & p) : PhysicsList(map, table_, chordFinderSetter_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  double timeLimit = p.getParameter<double>("MaxTrackTime")*ns;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "FTFP_BERT_XS_EML \n Flags for EM Physics "
			      << emPhys << ", for Hadronic Physics "
			      << hadPhys << " and tracking cut " << tracking
			      << "   t(ns)= " << timeLimit/ns;

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysicsXS(ver));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    RegisterPhysics(gn);
  }

  // Decays
  this->RegisterPhysics( new G4DecayPhysics(ver) );

  if (hadPhys) {
    G4HadronicProcessStore::Instance()->SetVerbose(ver);

    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysicsXS(ver));

    // Hadron Physics
    RegisterPhysics(  new G4HadronPhysicsFTFP_BERT(ver));

    // Stopping Physics
    RegisterPhysics( new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics(ver));

    RegisterPhysics( new G4NeutronCrossSectionXS(ver));

    // Neutron tracking cut
    if (tracking) {
      G4NeutronTrackingCut* ncut= new G4NeutronTrackingCut(ver);
      ncut->SetTimeLimit(timeLimit);
      RegisterPhysics(ncut);
    }
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,chordFinderSetter_,p));
}


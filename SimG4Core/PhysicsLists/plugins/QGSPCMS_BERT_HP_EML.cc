#include "QGSPCMS_BERT_HP_EML.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

#include "G4HadronPhysicsQGSP_BERT_HP.hh"

QGSPCMS_BERT_HP_EML::QGSPCMS_BERT_HP_EML(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
                              << "QGSP_BERT_HP_EML \n Flags for EM Physics " << emPhys << ", for Hadronic Physics "
                              << hadPhys << " and tracking cut " << tracking;

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new CMSEmStandardPhysics(ver));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    RegisterPhysics(gn);
  }

  // Decays
  this->RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    G4HadronicProcessStore::Instance()->SetVerbose(ver);

    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysicsHP(ver));

    // Hadron Physics
    RegisterPhysics(new G4HadronPhysicsQGSP_BERT_HP(ver));

    // Stopping Physics
    RegisterPhysics(new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics(new G4IonPhysics(ver));

    // Neutron tracking cut
    if (tracking) {
      RegisterPhysics(new G4NeutronTrackingCut(ver));
    }
  }
}

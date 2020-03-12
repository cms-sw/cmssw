#include "FTFPCMS_BERT_HP_EML.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsXS.h"
#include "SimG4Core/PhysicsLists/interface/CMSThermalNeutrons.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

#include "G4HadronPhysicsFTFP_BERT_HP.hh"

FTFPCMS_BERT_HP_EML::FTFPCMS_BERT_HP_EML(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  bool thermal = p.getUntrackedParameter<bool>("ThermalNeutrons");
  double timeLimit = p.getParameter<double>("MaxTrackTime") * CLHEP::ns;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
                              << "FTFP_BERT_HP_EML \n Flags for EM Physics " << emPhys << ", for Hadronic Physics "
                              << hadPhys << " and tracking cut " << tracking << "   t(ns)= " << timeLimit / CLHEP::ns
                              << " ThermalNeutrons: " << thermal;

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new CMSEmStandardPhysicsXS(ver));

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
    RegisterPhysics(new G4HadronPhysicsFTFP_BERT_HP(ver));

    // Stopping Physics
    RegisterPhysics(new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics(new G4IonPhysics(ver));

    // Neutron tracking cut
    if (tracking) {
      G4NeutronTrackingCut* ncut = new G4NeutronTrackingCut(ver);
      ncut->SetTimeLimit(timeLimit);
      RegisterPhysics(ncut);
    }
    if (thermal) {
      RegisterPhysics(new CMSThermalNeutrons(ver));
    }
  }
}

#include "FTFPCMS_BERT_EMZ.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

FTFPCMS_BERT_EMZ::FTFPCMS_BERT_EMZ(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  double timeLimit = p.getParameter<double>("MaxTrackTime") * CLHEP::ns;
  double minFTFP = p.getParameter<double>("EminFTFP") * CLHEP::GeV;
  double maxBERT = p.getParameter<double>("EmaxBERT") * CLHEP::GeV;
  double maxBERTpi = p.getParameter<double>("EmaxBERTpi") * CLHEP::GeV;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: FTFP_BERT_EMZ \n Flags for EM Physics " << emPhys
                              << ", for Hadronic Physics " << hadPhys << " and tracking cut " << tracking
                              << "   t(ns)= " << timeLimit / CLHEP::ns << "\n  transition energy Bertini/FTFP from "
                              << minFTFP / CLHEP::GeV << " to " << maxBERT / CLHEP::GeV << ":" << maxBERTpi / CLHEP::GeV
                              << " GeV";

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new G4EmStandardPhysics_option4(ver));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    RegisterPhysics(gn);
  }

  // Decays
  this->RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    G4HadronicProcessStore::Instance()->SetVerbose(ver);

    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics(ver));

    // Hadron Physics
    RegisterPhysics(new CMSHadronPhysicsFTFP_BERT(minFTFP, maxBERT, maxBERTpi));

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
  }
}

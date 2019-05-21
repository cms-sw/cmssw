#include "QGSPCMS_FTFP_BERT_EMZ.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

#include "G4DataQuestionaire.hh"
#include "G4HadronPhysicsQGSP_FTFP_BERT.hh"

QGSPCMS_FTFP_BERT_EMZ::QGSPCMS_FTFP_BERT_EMZ(const edm::ParameterSet& p) : PhysicsList(p) {
  G4DataQuestionaire it(photon);

  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  double timeLimit = p.getParameter<double>("MaxTrackTime") * CLHEP::ns;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
                              << "QGSP_FTFP_BERT_EMZ \n Flags for EM Physics " << emPhys << ", for Hadronic Physics "
                              << hadPhys << " and tracking cut " << tracking << "   t(ns)= " << timeLimit / CLHEP::ns;

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new G4EmStandardPhysics_option4(ver));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    RegisterPhysics(gn);
  }

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    G4HadronicProcessStore::Instance()->SetVerbose(ver);

    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics(ver));

    // Hadron Physics
    RegisterPhysics(new G4HadronPhysicsQGSP_FTFP_BERT(ver));

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

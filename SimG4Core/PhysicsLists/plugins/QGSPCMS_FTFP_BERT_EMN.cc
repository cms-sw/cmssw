#include "QGSPCMS_FTFP_BERT_EMN.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsXS.h"
#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSPCMS_FTFP_BERT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronicProcessStore.hh"

#include "G4HadronPhysicsQGSP_FTFP_BERT.hh"

QGSPCMS_FTFP_BERT_EMN::QGSPCMS_FTFP_BERT_EMN(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  double timeLimit = p.getParameter<double>("MaxTrackTime") * CLHEP::ns;
  double minFTFP = p.getParameter<double>("EminFTFP") * CLHEP::GeV;
  double maxBERT = p.getParameter<double>("EmaxBERT") * CLHEP::GeV;
  double minQGSP = p.getParameter<double>("EminQGSP") * CLHEP::GeV;
  double maxFTFP = p.getParameter<double>("EmaxFTFP") * CLHEP::GeV;
  double maxBERTpi = p.getParameter<double>("EmaxBERTpi") * CLHEP::GeV;
  edm::LogVerbatim("PhysicsList") << "You are using the simulation engine: "
                                  << "QGSP_FTFP_BERT_EMN \n Flags for EM Physics " << emPhys
                                  << ", for Hadronic Physics " << hadPhys << " and tracking cut " << tracking
                                  << "   t(ns)= " << timeLimit / CLHEP::ns << "\n  transition energy Bertini/FTFP from "
                                  << minFTFP / CLHEP::GeV << " to " << maxBERTpi / CLHEP::GeV << ":"
                                  << maxBERT / CLHEP::GeV << " GeV"
                                  << "\n  transition energy FTFP/QGSP from " << minQGSP / CLHEP::GeV << " to "
                                  << maxFTFP / CLHEP::GeV << " GeV";

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new CMSEmStandardPhysicsXS(ver, p));

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
    RegisterPhysics(new HadronPhysicsQGSPCMS_FTFP_BERT(minFTFP, maxBERT, minQGSP, maxFTFP, maxBERTpi));

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

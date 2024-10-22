#include "QBBCCMS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"

#include "G4EmStandardPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronicProcessStore.hh"

#include "G4HadronInelasticQBBC.hh"
#include "G4HadronElasticPhysicsXS.hh"
#include "G4IonPhysics.hh"
#include "G4NeutronTrackingCut.hh"

QBBCCMS::QBBCCMS(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  double timeLimit = p.getParameter<double>("MaxTrackTime") * CLHEP::ns;
  edm::LogVerbatim("PhysicsList") << "You are using the simulation engine: "
                                  << "FTFP_BERT_EMM: \n Flags for EM Physics: " << emPhys
                                  << "; Hadronic Physics: " << hadPhys << "; tracking cut: " << tracking
                                  << "; time limit(ns)= " << timeLimit / CLHEP::ns;

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new CMSEmStandardPhysics(ver, p));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    RegisterPhysics(gn);
  }

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    G4HadronicProcessStore::Instance()->SetVerbose(ver);

    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysicsXS(ver));

    // Hadron Physics
    RegisterPhysics(new G4HadronInelasticQBBC(ver));

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

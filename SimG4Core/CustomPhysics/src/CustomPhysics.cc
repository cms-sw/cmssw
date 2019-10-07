#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsListSS.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsLPM.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4HadronPhysicsFTFP_BERT.hh"
#include "G4SystemOfUnits.hh"

CustomPhysics::CustomPhysics(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool tracking = p.getParameter<bool>("TrackingCut");
  bool ssPhys = p.getUntrackedParameter<bool>("ExoticaPhysicsSS", false);
  double timeLimit = p.getParameter<double>("MaxTrackTime") * ns;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
                              << "FTFP_BERT_EMM for regular particles \n"
                              << "CustomPhysicsList " << ssPhys << " for exotics; "
                              << " tracking cut " << tracking << "  t(ns)= " << timeLimit / ns;
  // EM Physics
  RegisterPhysics(new CMSEmStandardPhysicsLPM(ver));

  // Synchroton Radiation & GN Physics
  RegisterPhysics(new G4EmExtraPhysics(ver));

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics(ver));

  // Hadron Physics
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT(ver));

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

  // Custom Physics
  if (ssPhys) {
    RegisterPhysics(new CustomPhysicsListSS("custom", p));
  } else {
    RegisterPhysics(new CustomPhysicsList("custom", p));
  }
}

#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsListSS.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/CustomPhysics/interface/APrimePhysics.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4SystemOfUnits.hh"

CustomPhysics::CustomPhysics(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool tracking = p.getParameter<bool>("TrackingCut");
  bool ssPhys = p.getUntrackedParameter<bool>("ExoticaPhysicsSS", false);
  bool dbrem = p.getUntrackedParameter<bool>("DBrem", false);
  double timeLimit = p.getParameter<double>("MaxTrackTime") * ns;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
                              << "FTFP_BERT_EMM for regular particles \n"
                              << "CustomPhysicsList " << ssPhys << " for exotics; "
                              << " tracking cut " << tracking << "  t(ns)= " << timeLimit / ns;
  // EM Physics
  RegisterPhysics(new CMSEmStandardPhysics(ver, p));

  // Synchroton Radiation & GN Physics
  RegisterPhysics(new G4EmExtraPhysics(ver));

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics(ver));

  // Hadron Physics
  RegisterPhysics(new CMSHadronPhysicsFTFP_BERT(ver));

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
  if (dbrem) {
    RegisterPhysics(new APrimePhysics(p.getUntrackedParameter<double>("DBremMass"),
                                      p.getUntrackedParameter<std::string>("DBremScaleFile"),
                                      p.getUntrackedParameter<double>("DBremBiasFactor")));
  } else if (ssPhys) {
    RegisterPhysics(new CustomPhysicsListSS("custom", p));
  } else {
    RegisterPhysics(new CustomPhysicsList("custom", p));
  }
}

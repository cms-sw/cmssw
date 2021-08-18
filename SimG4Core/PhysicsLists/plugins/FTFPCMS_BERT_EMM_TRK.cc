#include "FTFPCMS_BERT_EMM_TRK.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"

FTFPCMS_BERT_EMM_TRK::FTFPCMS_BERT_EMM_TRK(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  edm::LogVerbatim("PhysicsList") << "CMS Physics List FTFP_BERT_EMM_TRK";

  // EM physics
  RegisterPhysics(new CMSEmStandardPhysics(ver, p));

  // gamma and lepto-nuclear physics
  RegisterPhysics(new G4EmExtraPhysics(ver));

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  // Hadron elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics(ver));

  // Hadron inelastic physics
  RegisterPhysics(
      new CMSHadronPhysicsFTFP_BERT(3 * CLHEP::GeV, 6 * CLHEP::GeV, 12 * CLHEP::GeV, 2 * CLHEP::GeV, 4 * CLHEP::GeV));

  // Stopping physics
  RegisterPhysics(new G4StoppingPhysics(ver));

  // Ion physics
  RegisterPhysics(new G4IonPhysics(ver));
}

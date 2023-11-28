#include "FTFPCMS_BERT_EMMT.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsEMMT.h"
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"

FTFPCMS_BERT_EMMT::FTFPCMS_BERT_EMMT(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  double minFTFP = p.getParameter<double>("EminFTFP") * CLHEP::GeV;
  double maxBERT = p.getParameter<double>("EmaxBERT") * CLHEP::GeV;
  double maxBERTpi = p.getParameter<double>("EmaxBERTpi") * CLHEP::GeV;
  edm::LogVerbatim("PhysicsList") << "CMS Physics List FTFP_BERT_EMMT: "
                                  << "\n Flags for EM Physics: " << emPhys << "; Hadronic Physics: " << hadPhys
                                  << "\n Transition energy Bertini/FTFP from " << minFTFP / CLHEP::GeV << " to "
                                  << maxBERT / CLHEP::GeV << "; for pions to " << maxBERTpi / CLHEP::GeV << " GeV";

  if (emPhys) {
    // EM Physics + GN Physics (as part of specialized tracking)
    RegisterPhysics(new CMSEmStandardPhysicsEMMT(ver, p));
  }

  // Decays
  this->RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics(ver));

    // Hadron Physics
    RegisterPhysics(new CMSHadronPhysicsFTFP_BERT(minFTFP, maxBERT, maxBERTpi, minFTFP, maxBERT));

    // Stopping Physics
    RegisterPhysics(new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics(new G4IonPhysics(ver));
  }
}

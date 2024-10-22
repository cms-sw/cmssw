#include "FTFPCMS_BERT_EMN.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsXS.h"
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronicParameters.hh"

FTFPCMS_BERT_EMN::FTFPCMS_BERT_EMN(const edm::ParameterSet& p) : PhysicsList(p) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  double minFTFP = p.getParameter<double>("EminFTFP") * CLHEP::GeV;
  double maxBERT = p.getParameter<double>("EmaxBERT") * CLHEP::GeV;
  double maxBERTpi = p.getParameter<double>("EmaxBERTpi") * CLHEP::GeV;
  edm::LogVerbatim("PhysicsList") << "CMS Physics List FTFP_BERT_EMN: "
                                  << "\n Flags for EM Physics: " << emPhys << "; Hadronic Physics: " << hadPhys
                                  << "\n  transition energy Bertini/FTFP from " << minFTFP / CLHEP::GeV << " to "
                                  << maxBERT / CLHEP::GeV << ":" << maxBERTpi / CLHEP::GeV << " GeV";

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new CMSEmStandardPhysicsXS(ver, p));

    // Synchroton Radiation & GN Physics
    G4EmExtraPhysics* gn = new G4EmExtraPhysics(ver);
    RegisterPhysics(gn);
    bool mu = p.getParameter<bool>("G4MuonPairProductionByMuon");
    gn->MuonToMuMu(mu);
    edm::LogVerbatim("PhysicsList") << " Muon pair production by muons: " << mu;
  }

  // Decays
  this->RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    bool ngen = p.getParameter<bool>("G4NeutronGeneralProcess");
    bool bc = p.getParameter<bool>("G4BCHadronicProcess");
    bool hn = p.getParameter<bool>("G4LightHyperNucleiTracking");
    auto param = G4HadronicParameters::Instance();
    param->SetEnableNeutronGeneralProcess(ngen);
    param->SetEnableBCParticles(bc);
    param->SetEnableHyperNuclei(hn);
    edm::LogVerbatim("PhysicsList") << " Eneble neutron general process: " << ngen
                                    << "\n Enable b- and c- hadron physics: " << bc
                                    << "\n Enable light hyper-nuclei physics: " << hn;
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

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypHitFit.h"

TtSemiLepHypHitFit::TtSemiLepHypHitFit(const edm::ParameterSet& cfg)
    : TtSemiLepHypothesis(cfg),
      statusToken_(consumes<std::vector<int> >(cfg.getParameter<edm::InputTag>("status"))),
      partonsHadPToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsHadP"))),
      partonsHadQToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsHadQ"))),
      partonsHadBToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsHadB"))),
      partonsLepBToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsLepB"))),
      leptonsToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("leptons"))),
      neutrinosToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("neutrinos"))) {}

TtSemiLepHypHitFit::~TtSemiLepHypHitFit() {}

void TtSemiLepHypHitFit::buildHypo(edm::Event& evt,
                                   const edm::Handle<edm::View<reco::RecoCandidate> >& leps,
                                   const edm::Handle<std::vector<pat::MET> >& mets,
                                   const edm::Handle<std::vector<pat::Jet> >& jets,
                                   std::vector<int>& match,
                                   const unsigned int iComb) {
  edm::Handle<std::vector<int> > status;
  evt.getByToken(statusToken_, status);
  if ((*status)[iComb] != 0) {
    // create empty hypothesis if kinematic fit did not converge
    return;
  }

  edm::Handle<std::vector<pat::Particle> > partonsHadP;
  edm::Handle<std::vector<pat::Particle> > partonsHadQ;
  edm::Handle<std::vector<pat::Particle> > partonsHadB;
  edm::Handle<std::vector<pat::Particle> > partonsLepB;
  edm::Handle<std::vector<pat::Particle> > leptons;
  edm::Handle<std::vector<pat::Particle> > neutrinos;

  evt.getByToken(partonsHadPToken_, partonsHadP);
  evt.getByToken(partonsHadQToken_, partonsHadQ);
  evt.getByToken(partonsHadBToken_, partonsHadB);
  evt.getByToken(partonsLepBToken_, partonsLepB);
  evt.getByToken(leptonsToken_, leptons);
  evt.getByToken(neutrinosToken_, neutrinos);

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if (!(partonsHadP->empty() || partonsHadQ->empty() || partonsHadB->empty() || partonsLepB->empty())) {
    setCandidate(partonsHadP, iComb, lightQ_);
    setCandidate(partonsHadQ, iComb, lightQBar_);
    setCandidate(partonsHadB, iComb, hadronicB_);
    setCandidate(partonsLepB, iComb, leptonicB_);
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if (!leptons->empty()) {
    setCandidate(leptons, iComb, lepton_);
  }
  match.push_back(0);

  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if (!neutrinos->empty()) {
    setCandidate(neutrinos, iComb, neutrino_);
  }
}

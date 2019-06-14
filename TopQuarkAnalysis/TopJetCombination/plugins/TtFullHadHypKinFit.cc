#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullHadHypKinFit.h"

TtFullHadHypKinFit::TtFullHadHypKinFit(const edm::ParameterSet& cfg)
    : TtFullHadHypothesis(cfg),
      statusToken_(consumes<std::vector<int> >(cfg.getParameter<edm::InputTag>("status"))),
      lightQToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightQTag"))),
      lightQBarToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightQBarTag"))),
      bToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("bTag"))),
      bBarToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("bBarTag"))),
      lightPToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightPTag"))),
      lightPBarToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightPBarTag"))) {}

TtFullHadHypKinFit::~TtFullHadHypKinFit() {}

void TtFullHadHypKinFit::buildHypo(edm::Event& evt,
                                   const edm::Handle<std::vector<pat::Jet> >& jets,
                                   std::vector<int>& match,
                                   const unsigned int iComb) {
  edm::Handle<std::vector<int> > status;
  evt.getByToken(statusToken_, status);
  if ((*status)[iComb] != 0) {
    // create empty hypothesis if kinematic fit did not converge
    return;
  }

  edm::Handle<std::vector<pat::Particle> > lightQ;
  edm::Handle<std::vector<pat::Particle> > lightQBar;
  edm::Handle<std::vector<pat::Particle> > b;
  edm::Handle<std::vector<pat::Particle> > bBar;
  edm::Handle<std::vector<pat::Particle> > lightP;
  edm::Handle<std::vector<pat::Particle> > lightPBar;

  evt.getByToken(lightQToken_, lightQ);
  evt.getByToken(lightQBarToken_, lightQBar);
  evt.getByToken(bToken_, b);
  evt.getByToken(bBarToken_, bBar);
  evt.getByToken(lightPToken_, lightP);
  evt.getByToken(lightPBarToken_, lightPBar);

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if (!(lightQ->empty() || lightQBar->empty() || b->empty() || bBar->empty() || lightP->empty() ||
        lightPBar->empty())) {
    setCandidate(lightQ, iComb, lightQ_);
    setCandidate(lightQBar, iComb, lightQBar_);
    setCandidate(b, iComb, b_);
    setCandidate(bBar, iComb, bBar_);
    setCandidate(lightP, iComb, lightP_);
    setCandidate(lightPBar, iComb, lightPBar_);
  }
}

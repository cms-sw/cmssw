#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullHadHypothesis.h"

#include "DataFormats/PatCandidates/interface/Particle.h"

class TtFullHadHypKinFit : public TtFullHadHypothesis {
public:
  explicit TtFullHadHypKinFit(const edm::ParameterSet&);

private:
  /// build the event hypothesis key
  void buildKey() override { key_ = TtFullHadronicEvent::kKinFit; };
  /// build event hypothesis from the reco objects of a full-hadronic event
  void buildHypo(edm::Event&,
                 const edm::Handle<std::vector<pat::Jet> >&,
                 std::vector<int>&,
                 const unsigned int iComb) override;

  edm::EDGetTokenT<std::vector<int> > statusToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightQToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightQBarToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > bToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > bBarToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightPToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightPBarToken_;
};

TtFullHadHypKinFit::TtFullHadHypKinFit(const edm::ParameterSet& cfg)
    : TtFullHadHypothesis(cfg),
      statusToken_(consumes<std::vector<int> >(cfg.getParameter<edm::InputTag>("status"))),
      lightQToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightQTag"))),
      lightQBarToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightQBarTag"))),
      bToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("bTag"))),
      bBarToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("bBarTag"))),
      lightPToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightPTag"))),
      lightPBarToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("lightPBarTag"))) {}

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
    lightQ_ = makeCandidate(lightQ, iComb);
    lightQBar_ = makeCandidate(lightQBar, iComb);
    b_ = makeCandidate(b, iComb);
    bBar_ = makeCandidate(bBar, iComb);
    lightP_ = makeCandidate(lightP, iComb);
    lightPBar_ = makeCandidate(lightPBar, iComb);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtFullHadHypKinFit);

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

#include "DataFormats/PatCandidates/interface/Particle.h"

class TtSemiLepHypHitFit : public TtSemiLepHypothesis {
public:
  explicit TtSemiLepHypHitFit(const edm::ParameterSet&);

private:
  /// build the event hypothesis key
  void buildKey() override { key_ = TtSemiLeptonicEvent::kHitFit; };
  /// build event hypothesis from the reco objects of a semi-leptonic event
  void buildHypo(edm::Event&,
                 const edm::Handle<edm::View<reco::RecoCandidate> >&,
                 const edm::Handle<std::vector<pat::MET> >&,
                 const edm::Handle<std::vector<pat::Jet> >&,
                 std::vector<int>&,
                 const unsigned int iComb) override;

  edm::EDGetTokenT<std::vector<int> > statusToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsHadPToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsHadQToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsHadBToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsLepBToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > leptonsToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > neutrinosToken_;
};

TtSemiLepHypHitFit::TtSemiLepHypHitFit(const edm::ParameterSet& cfg)
    : TtSemiLepHypothesis(cfg),
      statusToken_(consumes<std::vector<int> >(cfg.getParameter<edm::InputTag>("status"))),
      partonsHadPToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsHadP"))),
      partonsHadQToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsHadQ"))),
      partonsHadBToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsHadB"))),
      partonsLepBToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("partonsLepB"))),
      leptonsToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("leptons"))),
      neutrinosToken_(consumes<std::vector<pat::Particle> >(cfg.getParameter<edm::InputTag>("neutrinos"))) {}

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
    lightQ_ = makeCandidate(partonsHadP, iComb);
    lightQBar_ = makeCandidate(partonsHadQ, iComb);
    hadronicB_ = makeCandidate(partonsHadB, iComb);
    leptonicB_ = makeCandidate(partonsLepB, iComb);
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if (!leptons->empty()) {
    lepton_ = makeCandidate(leptons, iComb);
  }
  match.push_back(0);

  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if (!neutrinos->empty()) {
    neutrino_ = makeCandidate(neutrinos, iComb);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtSemiLepHypHitFit);

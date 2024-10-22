#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullHadHypothesis.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

class TtFullHadHypGenMatch : public TtFullHadHypothesis {
public:
  explicit TtFullHadHypGenMatch(const edm::ParameterSet& cfg);

private:
  /// build the event hypothesis key
  void buildKey() override { key_ = TtFullHadronicEvent::kGenMatch; };
  /// build event hypothesis from the reco objects of a semi-leptonic event
  void buildHypo(edm::Event& evt,
                 const edm::Handle<std::vector<pat::Jet> >& jets,
                 std::vector<int>& match,
                 const unsigned int iComb) override;

protected:
  edm::EDGetTokenT<TtGenEvent> genEvtToken_;
};

TtFullHadHypGenMatch::TtFullHadHypGenMatch(const edm::ParameterSet& cfg)
    : TtFullHadHypothesis(cfg), genEvtToken_(consumes<TtGenEvent>(edm::InputTag("genEvt"))) {}

void TtFullHadHypGenMatch::buildHypo(edm::Event& evt,
                                     const edm::Handle<std::vector<pat::Jet> >& jets,
                                     std::vector<int>& match,
                                     const unsigned int iComb) {
  // -----------------------------------------------------
  // get genEvent (to distinguish between uds and c quarks)
  // -----------------------------------------------------
  edm::Handle<TtGenEvent> genEvt;
  evt.getByToken(genEvtToken_, genEvt);

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for (unsigned idx = 0; idx < match.size(); ++idx) {
    if (isValid(match[idx], jets)) {
      switch (idx) {
        case TtFullHadEvtPartons::LightQ:
          if (std::abs(genEvt->daughterQuarkOfWPlus()->pdgId()) == 4)
            lightQ_ = makeCandidate(jets, match[idx], jetCorrectionLevel("cQuark"));
          else
            lightQ_ = makeCandidate(jets, match[idx], jetCorrectionLevel("udsQuark"));
          break;
        case TtFullHadEvtPartons::LightQBar:
          if (std::abs(genEvt->daughterQuarkBarOfWPlus()->pdgId()) == 4)
            lightQBar_ = makeCandidate(jets, match[idx], jetCorrectionLevel("cQuark"));
          else
            lightQBar_ = makeCandidate(jets, match[idx], jetCorrectionLevel("udsQuark"));
          break;
        case TtFullHadEvtPartons::B:
          b_ = makeCandidate(jets, match[idx], jetCorrectionLevel("bQuark"));
          break;
        case TtFullHadEvtPartons::LightP:
          if (std::abs(genEvt->daughterQuarkOfWMinus()->pdgId()) == 4)
            lightP_ = makeCandidate(jets, match[idx], jetCorrectionLevel("cQuark"));
          else
            lightP_ = makeCandidate(jets, match[idx], jetCorrectionLevel("udsQuark"));
          break;
        case TtFullHadEvtPartons::LightPBar:
          if (std::abs(genEvt->daughterQuarkBarOfWMinus()->pdgId()) == 4)
            lightPBar_ = makeCandidate(jets, match[idx], jetCorrectionLevel("cQuark"));
          else
            lightPBar_ = makeCandidate(jets, match[idx], jetCorrectionLevel("udsQuark"));
          break;
        case TtFullHadEvtPartons::BBar:
          bBar_ = makeCandidate(jets, match[idx], jetCorrectionLevel("bQuark"));
          break;
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtFullHadHypGenMatch);

#ifndef TtFullLepHypGenMatch_h
#define TtFullLepHypGenMatch_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullLepHypothesis.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

class TtFullLepHypGenMatch : public TtFullLepHypothesis  {

 public:

  explicit TtFullLepHypGenMatch(const edm::ParameterSet&);
  ~TtFullLepHypGenMatch() override;

 private:

  /// build the event hypothesis key
  void buildKey() override { key_= TtFullLeptonicEvent::kGenMatch; };
  /// build event hypothesis from the reco objects of a semi-leptonic event
  void buildHypo(edm::Event& evt,
			 const edm::Handle<std::vector<pat::Electron > >& elecs,
			 const edm::Handle<std::vector<pat::Muon> >& mus,
			 const edm::Handle<std::vector<pat::Jet> >& jets,
			 const edm::Handle<std::vector<pat::MET> >& mets,
			 std::vector<int>& match,
			 const unsigned int iComb) override;

  template <typename O>
  int findMatchingLepton(const reco::GenParticle*,
			 const edm::Handle<std::vector<O> >&);
  void buildMatchingNeutrinos(edm::Event&,
                              const edm::Handle<std::vector<pat::MET> >&);

 protected:

  edm::EDGetTokenT<TtGenEvent> genEvtToken_;
};

#endif

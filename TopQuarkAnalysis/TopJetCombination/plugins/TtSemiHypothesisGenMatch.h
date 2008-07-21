#ifndef TtSemiHypothesisGenMatch_h
#define TtSemiHypothesisGenMatch_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiHypothesis.h"


class TtSemiHypothesisGenMatch : public TtSemiHypothesis  {

 public:

  explicit TtSemiHypothesisGenMatch(const edm::ParameterSet&);
  ~TtSemiHypothesisGenMatch();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiEvent::kGenMatch; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>& );
  int findMatchingLepton(edm::Event&, 
			 const edm::Handle<edm::View<reco::RecoCandidate> >&);
};

#endif

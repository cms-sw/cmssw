#ifndef TtSemiLepGenMatch_h
#define TtSemiLepGenMatch_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepGenMatch : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepGenMatch(const edm::ParameterSet&);
  ~TtSemiLepGenMatch();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kGenMatch; };  
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

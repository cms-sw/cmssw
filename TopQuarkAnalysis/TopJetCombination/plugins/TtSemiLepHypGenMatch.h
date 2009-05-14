#ifndef TtSemiLepHypGenMatch_h
#define TtSemiLepHypGenMatch_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypGenMatch : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypGenMatch(const edm::ParameterSet&);
  ~TtSemiLepHypGenMatch();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kGenMatch; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);
  int findMatchingLepton(edm::Event&, 
			 const edm::Handle<edm::View<reco::RecoCandidate> >&);
};

#endif

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
  /// find index of the candidate nearest to the singleLepton of the generator event in the collection; return -1 if this fails
  int findMatchingLepton(const edm::Handle<TtGenEvent>& genEvt, 
			 const edm::Handle<edm::View<reco::RecoCandidate> >&);

 private:

  int neutrinoSolutionType_;
};

#endif

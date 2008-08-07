#ifndef TtSemiHypothesisMaxSumPtWMass_h
#define TtSemiHypothesisMaxSumPtWMass_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiHypothesis.h"


class TtSemiHypothesisMaxSumPtWMass : public TtSemiHypothesis  {

 public:

  explicit TtSemiHypothesisMaxSumPtWMass(const edm::ParameterSet&);
  ~TtSemiHypothesisMaxSumPtWMass();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kMaxSumPtWMass; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>& );

 private:

  unsigned maxNJets_;
};

#endif

#ifndef TtSemiLepMaxSumPtWMass_h
#define TtSemiLepMaxSumPtWMass_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepMaxSumPtWMass : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepMaxSumPtWMass(const edm::ParameterSet&);
  ~TtSemiLepMaxSumPtWMass();

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
  double wMass_;
};

#endif

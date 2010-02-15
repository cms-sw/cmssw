#ifndef TtSemiLepHypMaxSumPtWMass_h
#define TtSemiLepHypMaxSumPtWMass_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypMaxSumPtWMass : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypMaxSumPtWMass(const edm::ParameterSet&);
  ~TtSemiLepHypMaxSumPtWMass();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kMaxSumPtWMass; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);

 private:

  int maxNJets_;
  double wMass_;
};

#endif

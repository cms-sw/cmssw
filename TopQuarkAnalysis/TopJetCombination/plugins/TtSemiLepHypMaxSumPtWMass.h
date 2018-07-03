#ifndef TtSemiLepHypMaxSumPtWMass_h
#define TtSemiLepHypMaxSumPtWMass_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypMaxSumPtWMass : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypMaxSumPtWMass(const edm::ParameterSet& cfg): TtSemiLepHypothesis(cfg) {};
  ~TtSemiLepHypMaxSumPtWMass() override {};

 private:

  /// build the event hypothesis key
  void buildKey() override { key_= TtSemiLeptonicEvent::kMaxSumPtWMass; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  void buildHypo(edm::Event& evt,
			 const edm::Handle<edm::View<reco::RecoCandidate> >& leps,
			 const edm::Handle<std::vector<pat::MET> >& mets,
			 const edm::Handle<std::vector<pat::Jet> >& jets,
			 std::vector<int>& match, const unsigned int iComb) override { TtSemiLepHypothesis::buildHypo(leps, mets, jets, match); };
};

#endif

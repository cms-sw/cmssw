#ifndef TtSemiLepHypWMassMaxSumPt_h
#define TtSemiLepHypWMassMaxSumPt_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypWMassMaxSumPt : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypWMassMaxSumPt(const edm::ParameterSet&cfg): TtSemiLepHypothesis(cfg) {};
  ~TtSemiLepHypWMassMaxSumPt() {};

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kWMassMaxSumPt; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event& evt,
			 const edm::Handle<edm::View<reco::RecoCandidate> >& leps,
			 const edm::Handle<std::vector<pat::MET> >& mets,
			 const edm::Handle<std::vector<pat::Jet> >& jets,
			 std::vector<int>& match, const unsigned int iComb) { TtSemiLepHypothesis::buildHypo(leps, mets, jets, match); };
};

#endif

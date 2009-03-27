#ifndef TtSemiLepHypWMassMaxSumPt_h
#define TtSemiLepHypWMassMaxSumPt_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypWMassMaxSumPt : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypWMassMaxSumPt(const edm::ParameterSet&);
  ~TtSemiLepHypWMassMaxSumPt();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kWMassMaxSumPt; };  
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

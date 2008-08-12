#ifndef TtSemiLepWMassMaxSumPt_h
#define TtSemiLepWMassMaxSumPt_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepWMassMaxSumPt : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepWMassMaxSumPt(const edm::ParameterSet&);
  ~TtSemiLepWMassMaxSumPt();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kWMassMaxSumPt; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&);

 private:

  unsigned maxNJets_;
  double wMass_;
};

#endif

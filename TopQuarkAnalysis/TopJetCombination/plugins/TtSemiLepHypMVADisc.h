#ifndef TtSemiLepHypMVADisc_h
#define TtSemiLepHypMVADisc_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypMVADisc : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypMVADisc(const edm::ParameterSet&);
  ~TtSemiLepHypMVADisc();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kMVADisc; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);
};

#endif

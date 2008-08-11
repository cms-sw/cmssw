#ifndef TtSemiLepMVADisc_h
#define TtSemiLepMVADisc_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepMVADisc : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepMVADisc(const edm::ParameterSet&);
  ~TtSemiLepMVADisc();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kMVADisc; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>& );
};

#endif

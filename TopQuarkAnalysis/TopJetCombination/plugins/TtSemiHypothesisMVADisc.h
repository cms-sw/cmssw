#ifndef TtSemiHypothesisMVADisc_h
#define TtSemiHypothesisMVADisc_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiHypothesis.h"


class TtSemiHypothesisMVADisc : public TtSemiHypothesis  {

 public:

  explicit TtSemiHypothesisMVADisc(const edm::ParameterSet&);
  ~TtSemiHypothesisMVADisc();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiEvent::kMVADisc; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 const edm::Handle<std::vector<int> >& );
};

#endif

#ifndef TtSemiLepHypGeom_h
#define TtSemiLepHypGeom_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypGeom : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypGeom(const edm::ParameterSet&);
  ~TtSemiLepHypGeom();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kGeom; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);

};

#endif

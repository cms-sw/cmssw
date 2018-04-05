#ifndef TtSemiLepHypGeom_h
#define TtSemiLepHypGeom_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypGeom : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypGeom(const edm::ParameterSet& cfg): TtSemiLepHypothesis(cfg) {};
  ~TtSemiLepHypGeom() override {};

 private:

  /// build the event hypothesis key
  void buildKey() override { key_= TtSemiLeptonicEvent::kGeom; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  void buildHypo(edm::Event& evt,
			 const edm::Handle<edm::View<reco::RecoCandidate> >& leps,
			 const edm::Handle<std::vector<pat::MET> >& mets,
			 const edm::Handle<std::vector<pat::Jet> >& jets,
			 std::vector<int>& match, const unsigned int iComb) override { TtSemiLepHypothesis::buildHypo(leps, mets, jets, match); };
};

#endif

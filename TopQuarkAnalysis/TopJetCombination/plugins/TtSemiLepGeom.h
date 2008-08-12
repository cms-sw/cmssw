#ifndef TtSemiLepGeom_h
#define TtSemiLepGeom_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepGeom : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepGeom(const edm::ParameterSet&);
  ~TtSemiLepGeom();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kGeom; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&);
  double distance(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);

 private:

  int maxNJets_;
  bool useDeltaR_;
};

#endif

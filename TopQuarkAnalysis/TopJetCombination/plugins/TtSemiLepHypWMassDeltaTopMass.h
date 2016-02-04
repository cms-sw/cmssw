#ifndef TtSemiLepHypWMassDeltaTopMass_h
#define TtSemiLepHypWMassDeltaTopMass_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


class TtSemiLepHypWMassDeltaTopMass : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypWMassDeltaTopMass(const edm::ParameterSet&);
  ~TtSemiLepHypWMassDeltaTopMass();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kWMassDeltaTopMass; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);

 private:

  int maxNJets_;
  double wMass_;
  bool useBTagging_;
  std::string bTagAlgorithm_;
  double minBDiscBJets_;
  double maxBDiscLightJets_;
  int neutrinoSolutionType_;
};

#endif

#ifndef TtSemiLepHypHitFit_h
#define TtSemiLepHypHitFit_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

class TtSemiLepHypHitFit : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypHitFit(const edm::ParameterSet&);
  ~TtSemiLepHypHitFit();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kHitFit; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);

  edm::InputTag status_;
  edm::InputTag partonsHadP_;
  edm::InputTag partonsHadQ_;
  edm::InputTag partonsHadB_;
  edm::InputTag partonsLepB_;
  edm::InputTag leptons_;
  edm::InputTag neutrinos_;

};

#endif

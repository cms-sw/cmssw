#ifndef TtSemiLepHypKinFit_h
#define TtSemiLepHypKinFit_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

class TtSemiLepHypKinFit : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypKinFit(const edm::ParameterSet&);
  ~TtSemiLepHypKinFit();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtSemiLeptonicEvent::kKinFit; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&);

  edm::InputTag status_;
  edm::InputTag partons_;
  edm::InputTag leptons_;
  edm::InputTag neutrinos_;

};

#endif

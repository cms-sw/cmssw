#ifndef TtSemiLepKinFit_h
#define TtSemiLepKinFit_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

class TtSemiLepKinFit : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepKinFit(const edm::ParameterSet&);
  ~TtSemiLepKinFit();

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

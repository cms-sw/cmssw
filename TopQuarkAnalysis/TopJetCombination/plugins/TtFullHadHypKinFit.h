#ifndef TtFullHadHypKinFit_h
#define TtFullHadHypKinFit_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullHadHypothesis.h"

class TtFullHadHypKinFit : public TtFullHadHypothesis  {

 public:

  explicit TtFullHadHypKinFit(const edm::ParameterSet&);
  ~TtFullHadHypKinFit();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtFullHadronicEvent::kKinFit; };  
  /// build event hypothesis from the reco objects of a full-hadronic event 
  virtual void buildHypo(edm::Event&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb);

  edm::InputTag status_;
  edm::InputTag lightQTag_;
  edm::InputTag lightQBarTag_;
  edm::InputTag bTag_;
  edm::InputTag bBarTag_;
  edm::InputTag lightPTag_;
  edm::InputTag lightPBarTag_;

};

#endif

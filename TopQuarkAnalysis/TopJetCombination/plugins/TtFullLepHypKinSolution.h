#ifndef TtFullLepHypKinSolution_h
#define TtFullLepHypKinSolution_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullLepHypothesis.h"

class TtFullLepHypKinSolution : public TtFullLepHypothesis  {

 public:

  explicit TtFullLepHypKinSolution(const edm::ParameterSet&);
  ~TtFullLepHypKinSolution();

 private:
  
  /// build the event hypothesis key
  virtual void buildKey() { key_= TtEvent::kKinSolution; };  
  /// build event hypothesis from the reco objects of a full-leptonic event 
  virtual void buildHypo(edm::Event& evt,
			 const edm::Handle<std::vector<pat::Electron > >& elecs, 
			 const edm::Handle<std::vector<pat::Muon> >& mus, 
			 const edm::Handle<std::vector<pat::Jet> >& jets, 
			 const edm::Handle<std::vector<pat::MET> >& mets, 
			 std::vector<int>& match,
			 const unsigned int iComb);

  edm::InputTag particleIdcs_;  
  edm::InputTag nus_;
  edm::InputTag nuBars_;  
  edm::InputTag solWeight_; 
  
};

#endif

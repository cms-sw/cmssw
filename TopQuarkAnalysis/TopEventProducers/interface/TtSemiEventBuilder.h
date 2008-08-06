#ifndef TtSemiEventBuilder_h
#define TtSemiEventBuilder_h

#include <memory>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"


class TtSemiEventBuilder : public edm::EDProducer {

 public:

  explicit TtSemiEventBuilder(const edm::ParameterSet&);
  ~TtSemiEventBuilder();
  
 private:

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  // hypothesis
  std::vector<edm::InputTag> hyps_;
  std::vector<edm::InputTag> keys_;  

  // meta information
  std::vector<edm::InputTag> matches_;  

  // kinFit extras
  edm::ParameterSet kinFit_;
  edm::InputTag fitChi2_;
  edm::InputTag fitProb_;

  // gen match extras
  int decay_;
  edm::InputTag genEvt_;

  edm::ParameterSet genMatch_;
  edm::InputTag sumPt_;
  edm::InputTag sumDR_;

  // mvaDisc extras
  edm::ParameterSet mvaDisc_;
  edm::InputTag meth_;
  edm::InputTag disc_;
};

#endif

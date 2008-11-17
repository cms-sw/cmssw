#ifndef TtSemiLepEvtBuilder_h
#define TtSemiLepEvtBuilder_h

#include <memory>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"


class TtSemiLepEvtBuilder : public edm::EDProducer {

 public:

  explicit TtSemiLepEvtBuilder(const edm::ParameterSet&);
  ~TtSemiLepEvtBuilder();
  
 private:

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  int verbosity_;

  // hypotheses
  std::vector<std::string> hyps_;

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

#ifndef TauAnalysis_MCEmbeddingTools_PileUpWeightProducer_h
#define TauAnalysis_MCEmbeddingTools_PileUpWeightProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <TH1.h>

#include <string>

class PileUpWeightProducer : public edm::EDProducer
{
 public:
  explicit PileUpWeightProducer(const edm::ParameterSet&);
  ~PileUpWeightProducer();

  void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag srcPileUpSummaryInfo_;
  TH1* weightHisto_;
};

#endif


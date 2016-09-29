#include "TauAnalysis/MCEmbeddingTools/plugins/CandViewCountEventSelFlagProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CandViewCountEventSelFlagProducer::CandViewCountEventSelFlagProducer(const edm::ParameterSet& cfg)
  : eventSelector_(cfg, consumesCollector())
{ 
  produces<bool>();
}

void CandViewCountEventSelFlagProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  evt.put(std::make_unique<bool>(eventSelector_(evt, es)));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CandViewCountEventSelFlagProducer);

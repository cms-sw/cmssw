#include "TauAnalysis/MCEmbeddingTools/plugins/CandViewCountEventSelFlagProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CandViewCountEventSelFlagProducer::CandViewCountEventSelFlagProducer(const edm::ParameterSet& cfg)
  : eventSelector_(cfg)
{ 
  produces<bool>();
}

void CandViewCountEventSelFlagProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<bool> boolPtr(new bool(eventSelector_(evt, es)));
  evt.put(boolPtr);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CandViewCountEventSelFlagProducer);

#include "TauAnalysis/MCEmbeddingTools/plugins/DummyBoolEventSelFlagProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

DummyBoolEventSelFlagProducer::DummyBoolEventSelFlagProducer(const edm::ParameterSet& cfg)
{ 
  produces<bool>();
}

DummyBoolEventSelFlagProducer::~DummyBoolEventSelFlagProducer()
{
// nothing to be done yet...
}

void DummyBoolEventSelFlagProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<bool> boolPtr(new bool(true));
  evt.put(boolPtr);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DummyBoolEventSelFlagProducer);

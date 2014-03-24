#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

class SeedingLayersEDProducer: public edm::EDProducer {
public:
  SeedingLayersEDProducer(const edm::ParameterSet& iConfig);
  ~SeedingLayersEDProducer();

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  SeedingLayerSetsBuilder builder_;
};

SeedingLayersEDProducer::SeedingLayersEDProducer(const edm::ParameterSet& iConfig):
  builder_(iConfig, consumesCollector())
{
  produces<SeedingLayerSetsHits>();
}
SeedingLayersEDProducer::~SeedingLayersEDProducer() {}

void SeedingLayersEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if(builder_.check(iSetup)) {
    builder_.updateEventSetup(iSetup);
  }

  // Get hits
  std::auto_ptr<SeedingLayerSetsHits> prod(new SeedingLayerSetsHits(builder_.numberOfLayersInSet(),
                                                                    &builder_.layerSetIndices(),
                                                                    &builder_.layerNames(),
                                                                    builder_.layerDets()));
  std::pair<std::vector<unsigned int>, ctfseeding::SeedingLayer::Hits> idxHits = builder_.hits(iEvent, iSetup);
  prod->swapHits(idxHits.first, idxHits.second);
  //prod->print();

  iEvent.put(prod);
}

DEFINE_FWK_MODULE(SeedingLayersEDProducer);

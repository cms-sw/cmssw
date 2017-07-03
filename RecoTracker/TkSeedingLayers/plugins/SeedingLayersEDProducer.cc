#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class dso_hidden SeedingLayersEDProducer: public edm::stream::EDProducer<> {
public:
  SeedingLayersEDProducer(const edm::ParameterSet& iConfig);
  ~SeedingLayersEDProducer() override;

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
  auto prod = std::make_unique<SeedingLayerSetsHits>(builder_.numberOfLayersInSet(),
                                                    &builder_.layerSetIndices(),
                                                    &builder_.layerNames(),
                                                     builder_.layerDets());
  std::vector<unsigned int> idx; ctfseeding::SeedingLayer::Hits hits; 
  builder_.hits(iEvent, iSetup,idx,hits);
  hits.shrink_to_fit();
  prod->swapHits(idx,hits);
  //prod->print();

  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(SeedingLayersEDProducer);

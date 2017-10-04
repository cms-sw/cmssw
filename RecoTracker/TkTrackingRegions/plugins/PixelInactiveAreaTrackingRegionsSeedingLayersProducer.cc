#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "AreaSeededTrackingRegionsBuilder.h"

#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionsSeedingLayerSetsHits.h"

#include "PixelInactiveAreaFinder.h"

#include <vector>
#include <utility>
#include <memory>

class PixelInactiveAreaTrackingRegionsSeedingLayersProducer: public edm::stream::EDProducer<> {
public:
  using ProductType = TrackingRegionsSeedingLayerSetsHits;

  PixelInactiveAreaTrackingRegionsSeedingLayersProducer(const edm::ParameterSet& iConfig);
  ~PixelInactiveAreaTrackingRegionsSeedingLayersProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  SeedingLayerSetsBuilder seedingLayerSetsBuilder_;
  PixelInactiveAreaFinder inactiveAreaFinder_;
  AreaSeededTrackingRegionsBuilder trackingRegionsBuilder_;
};

PixelInactiveAreaTrackingRegionsSeedingLayersProducer::PixelInactiveAreaTrackingRegionsSeedingLayersProducer(const edm::ParameterSet& iConfig):
  seedingLayerSetsBuilder_(iConfig, consumesCollector()),
  inactiveAreaFinder_(iConfig, seedingLayerSetsBuilder_.layers(), seedingLayerSetsBuilder_.layerSetIndices()),
  trackingRegionsBuilder_(iConfig.getParameter<edm::ParameterSet>("RegionPSet"), consumesCollector())
{
  produces<ProductType>();
}

void PixelInactiveAreaTrackingRegionsSeedingLayersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription descRegion;
  AreaSeededTrackingRegionsBuilder::fillDescriptions(descRegion);
  desc.add<edm::ParameterSetDescription>("RegionPSet", descRegion);

  PixelInactiveAreaFinder::fillDescriptions(desc);
  SeedingLayerSetsBuilder::fillDescriptions(desc);

  descriptions.add("pixelInactiveAreaTrackingRegionsAndSeedingLayers", desc);
}

void PixelInactiveAreaTrackingRegionsSeedingLayersProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto ret = std::make_unique<ProductType>();

  auto builder = trackingRegionsBuilder_.beginEvent(iEvent, iSetup);

  const auto& areas = inactiveAreaFinder_.inactiveAreas(iEvent, iSetup);
  ret->reserve(areas.size());
  for(const auto& inactiveArea: areas) {
    ret->emplace_back(builder.regions(inactiveArea.areas()), SeedingLayerSetsHits());
#ifdef NOT_IMPLEMENTED_YET
                      seedingLayerSetsBuilder_.hits(inactiveAreas.layers(), iEvent, iSetup));
#endif
  }

  iEvent.put(std::move(ret));
}

DEFINE_FWK_MODULE(PixelInactiveAreaTrackingRegionsSeedingLayersProducer);

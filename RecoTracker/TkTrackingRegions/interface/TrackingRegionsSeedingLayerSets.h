#ifndef RecoTracker_TkTrackingRegions_TrackingRegionsSeedingLayerSets_H
#define RecoTracker_TkTrackingRegions_TrackingRegionsSeedingLayerSets_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include <vector>
#include <memory>

class TrackingRegionsSeedingLayerSets {
public:
  TrackingRegionsSeedingLayerSets() = default;
  explicit TrackingRegionsSeedingLayerSets(const SeedingLayerSetsHits *seedingLayerSetsHits):
    seedingLayerSetsHits_(seedingLayerSetsHits)
  {}
  ~TrackingRegionsSeedingLayerSets() = default;
  TrackingRegionsSeedingLayerSets(TrackingRegionsSeedingLayerSets const&)=delete;
  TrackingRegionsSeedingLayerSets& operator=(TrackingRegionsSeedingLayerSets const&)=delete;
  TrackingRegionsSeedingLayerSets(TrackingRegionsSeedingLayerSets &&)=default;
  TrackingRegionsSeedingLayerSets& operator=(TrackingRegionsSeedingLayerSets &&)=default;

  void reserve(size_t s) { regionLayers_.reserve(s); }

  // layerSets index points to a layer set within seedingLayerSetsHits_
  void emplace_back(std::unique_ptr<TrackingRegion>&& region,
                    std::vector<SeedingLayerSetsHits::LayerSetIndex>&& layerSets) {
    regionLayers_.emplace_back(std::move(region), std::move(layerSets));
  }

  void swap(TrackingRegionsSeedingLayerSets& other) {
    std::swap(seedingLayerSetsHits_, other.seedingLayerSetsHits_);
    regionLayers_.swap(other.regionLayers_);
  }

private:
  class Element {
  public:
    Element(std::unique_ptr<TrackingRegion>&& region, std::vector<SeedingLayerSetsHits::LayerSetIndex>&& layerSets):
      region_(std::move(region)), layerSets_(std::move(layerSets)) {}
    ~Element() = default;
    Element(Element const&)=delete;
    Element& operator=(Element const&)=delete;
    Element(Element &&)=default;
    Element& operator=(Element &&)=default;

    std::unique_ptr<TrackingRegion> region_;
    std::vector<SeedingLayerSetsHits::LayerSetIndex> layerSets_;
  };

  const SeedingLayerSetsHits *seedingLayerSetsHits_ = nullptr;
  std::vector<Element> regionLayers_;
};

#endif

#ifndef RecoTracker_TkTrackingRegions_TrackingRegionsSeedingLayerSetsHits_H
#define RecoTracker_TkTrackingRegions_TrackingRegionsSeedingLayerSetsHits_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include <vector>
#include <memory>

class TrackingRegionsSeedingLayerSetsHits {
public:
  TrackingRegionsSeedingLayerSetsHits() = default;
  ~TrackingRegionsSeedingLayerSetsHits() = default;
  TrackingRegionsSeedingLayerSetsHits(TrackingRegionsSeedingLayerSetsHits const&)=delete;
  TrackingRegionsSeedingLayerSetsHits& operator=(TrackingRegionsSeedingLayerSetsHits const&)=delete;
  TrackingRegionsSeedingLayerSetsHits(TrackingRegionsSeedingLayerSetsHits &&)=default;
  TrackingRegionsSeedingLayerSetsHits& operator=(TrackingRegionsSeedingLayerSetsHits &&)=default;

  void reserve(size_t s) { regionLayers_.reserve(s); }

  void emplace_back(std::vector<std::unique_ptr<TrackingRegion> >&& regions, SeedingLayerSetsHits&& layers) {
    regionLayers_.emplace_back(std::move(regions), std::move(layers));
  }

  void swap(TrackingRegionsSeedingLayerSetsHits& other) {
    regionLayers_.swap(other.regionLayers_);
  }

private:
  class Element {
  public:
    Element(std::vector<std::unique_ptr<TrackingRegion> >&& regions, SeedingLayerSetsHits&& layers):
      regions_(std::move(regions)), layers_(std::move(layers)) {}
    ~Element() = default;
    Element(Element const&)=delete;
    Element& operator=(Element const&)=delete;
    Element(Element &&)=default;
    Element& operator=(Element &&)=default;

    std::vector<std::unique_ptr<TrackingRegion> > regions_;
    SeedingLayerSetsHits layers_;
  };
  std::vector<Element> regionLayers_;
};

#endif

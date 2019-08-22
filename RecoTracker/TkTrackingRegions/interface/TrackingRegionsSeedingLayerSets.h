#ifndef RecoTracker_TkTrackingRegions_TrackingRegionsSeedingLayerSets_H
#define RecoTracker_TkTrackingRegions_TrackingRegionsSeedingLayerSets_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include <vector>
#include <memory>

class TrackingRegionsSeedingLayerSets {
  class Element;

public:
  class RegionLayers {
  public:
    using SeedingLayerSet = SeedingLayerSetsHits::SeedingLayerSet;

    RegionLayers(const Element* elem, const SeedingLayerSetsHits* seedingLayerSetsHits)
        : elem_(elem), seedingLayerSetsHits_(seedingLayerSetsHits) {}

    const TrackingRegion& region() const;
    std::vector<SeedingLayerSet> layerPairs() const;

  private:
    const Element* elem_;
    const SeedingLayerSetsHits* seedingLayerSetsHits_;
  };

  class const_iterator {
  public:
    using internal_iterator_type = std::vector<Element>::const_iterator;
    using value_type = RegionLayers;
    using difference_type = internal_iterator_type::difference_type;

    const_iterator(internal_iterator_type iter, const SeedingLayerSetsHits* seedingLayerSetsHits)
        : iter_(iter), seedingLayerSetsHits_(seedingLayerSetsHits) {}

    value_type operator*() const { return RegionLayers(&(*iter_), seedingLayerSetsHits_); }
    const_iterator& operator++() {
      ++iter_;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator clone(*this);
      ++(*this);
      return clone;
    }

    bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }

  private:
    internal_iterator_type iter_;
    const SeedingLayerSetsHits* seedingLayerSetsHits_;
  };

  TrackingRegionsSeedingLayerSets() = default;
  explicit TrackingRegionsSeedingLayerSets(const SeedingLayerSetsHits* seedingLayerSetsHits)
      : seedingLayerSetsHits_(seedingLayerSetsHits) {}
  ~TrackingRegionsSeedingLayerSets() = default;
  TrackingRegionsSeedingLayerSets(TrackingRegionsSeedingLayerSets const&) = delete;
  TrackingRegionsSeedingLayerSets& operator=(TrackingRegionsSeedingLayerSets const&) = delete;
  TrackingRegionsSeedingLayerSets(TrackingRegionsSeedingLayerSets&&) = default;
  TrackingRegionsSeedingLayerSets& operator=(TrackingRegionsSeedingLayerSets&&) = default;

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

  const SeedingLayerSetsHits& seedingLayerSetsHits() const { return *seedingLayerSetsHits_; }
  size_t regionsSize() const { return regionLayers_.size(); }

  const_iterator begin() const { return const_iterator(regionLayers_.begin(), seedingLayerSetsHits_); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(regionLayers_.end(), seedingLayerSetsHits_); }
  const_iterator cend() const { return end(); }

private:
  class Element {
  public:
    Element(std::unique_ptr<TrackingRegion>&& region, std::vector<SeedingLayerSetsHits::LayerSetIndex>&& layerSets)
        : region_(std::move(region)), layerSets_(std::move(layerSets)) {}
    ~Element() = default;
    Element(Element const&) = delete;
    Element& operator=(Element const&) = delete;
    Element(Element&&) = default;
    Element& operator=(Element&&) = default;

    const TrackingRegion& region() const { return *region_; }
    const std::vector<SeedingLayerSetsHits::LayerSetIndex>& layerSets() const { return layerSets_; }

  private:
    std::unique_ptr<TrackingRegion> region_;
    std::vector<SeedingLayerSetsHits::LayerSetIndex> layerSets_;
  };

  const SeedingLayerSetsHits* seedingLayerSetsHits_ = nullptr;
  std::vector<Element> regionLayers_;
};

inline const TrackingRegion& TrackingRegionsSeedingLayerSets::RegionLayers::region() const { return elem_->region(); }

inline std::vector<TrackingRegionsSeedingLayerSets::RegionLayers::SeedingLayerSet>
TrackingRegionsSeedingLayerSets::RegionLayers::layerPairs() const {
  std::vector<SeedingLayerSet> ret;  // TODO: get rid of the vector with more boilerplate code (sigh)

  const auto& layerSets = elem_->layerSets();
  ret.reserve(layerSets.size());
  for (const auto& ind : layerSets) {
    ret.push_back((*seedingLayerSetsHits_)[ind]);
  }
  return ret;
}

#endif

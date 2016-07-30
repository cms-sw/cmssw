#ifndef RecoTracker_TkHitPairs_IntermediateHitDoublets_h
#define RecoTracker_TkHitPairs_IntermediateHitDoublets_h

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

namespace ihd {
  class RegionIndex {
  public:
    RegionIndex(const TrackingRegion *reg, unsigned int ind):
      region_(reg),
      layerSetBeginIndex_(ind),
      layerSetEndIndex_(ind)
    {}

    void setLayerSetsEnd(unsigned int end) { layerSetEndIndex_ = end; }

    const TrackingRegion& region() const { return *region_; }
    unsigned int layerSetBeginIndex() const { return layerSetBeginIndex_; }
    unsigned int layerSetEndIndex() const { return layerSetEndIndex_; }

  private:
    const TrackingRegion *region_;
    unsigned int layerSetBeginIndex_; /// index to doublets_, pointing to the beginning of the layer pairs of this region
    unsigned int layerSetEndIndex_;   /// index to doublets_, pointing to the end of the layer pairs of this region
  };

  template <typename T>
  class RegionLayerHits {
  public:
    using const_iterator = typename std::vector<T>::const_iterator;

    // Taking T* to have compatible interface with IntermediateHitTriplets::RegionLayerHits
    template <typename TMP>
    RegionLayerHits(const TrackingRegion* region, const TMP*, const_iterator begin, const_iterator end):
      region_(region), layerSetsBegin_(begin), layerSetsEnd_(end) {}

    const TrackingRegion& region() const { return *region_; }

    const_iterator begin() const { return layerSetsBegin_; }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return layerSetsEnd_; }
    const_iterator cend() const { return end(); }

  private:
    const TrackingRegion *region_;
    const const_iterator layerSetsBegin_;
    const const_iterator layerSetsEnd_;
  };

  template<typename ValueType, typename HitSetType>
  class const_iterator {
  public:
    using internal_iterator_type = typename std::vector<RegionIndex>::const_iterator;
    using value_type = ValueType;
    using difference_type = internal_iterator_type::difference_type;

    const_iterator(const HitSetType *hst, internal_iterator_type iter): hitSets_(hst), iter_(iter) {}

    value_type operator*() const {
      return value_type(&(iter_->region()),
                        hitSets_,
                        hitSets_->layerSetsBegin() + iter_->layerSetBeginIndex(),
                        hitSets_->layerSetsBegin() + iter_->layerSetEndIndex());
    }

    const_iterator& operator++() { ++iter_; return *this; }
    const_iterator operator++(int) {
      const_iterator clone(*this);
      ++iter_;
      return clone;
    }

    bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }

  private:
    const HitSetType *hitSets_;
    internal_iterator_type iter_;
  };
}

/**
 * Simple container of temporary information delivered from hit pair
 * generator to hit triplet generator via edm::Event.
 */
class IntermediateHitDoublets {
public:
  using LayerPair = std::tuple<SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex>;
  using RegionIndex = ihd::RegionIndex;

  class LayerPairHitDoublets {
  public:
    LayerPairHitDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets, LayerHitMapCache&& cache):
      layerPair_(layerSet[0].index(), layerSet[1].index()),
      doublets_(std::move(doublets)),
      cache_(std::move(cache))
    {}

    const LayerPair& layerPair() const { return layerPair_; }
    SeedingLayerSetsHits::LayerIndex innerLayerIndex() const { return std::get<0>(layerPair_); }
    SeedingLayerSetsHits::LayerIndex outerLayerIndex() const { return std::get<1>(layerPair_); }

    const HitDoublets& doublets() const { return doublets_; }
    const LayerHitMapCache& cache() const { return cache_; }

  private:
    LayerPair layerPair_;
    HitDoublets doublets_;
    LayerHitMapCache cache_;
  };

  ////////////////////

  using RegionLayerHits = ihd::RegionLayerHits<LayerPairHitDoublets>;

  ////////////////////

  using const_iterator = ihd::const_iterator<RegionLayerHits, IntermediateHitDoublets>;

  ////////////////////

  IntermediateHitDoublets(): seedingLayers_(nullptr) {}
  explicit IntermediateHitDoublets(const SeedingLayerSetsHits *seedingLayers): seedingLayers_(seedingLayers) {}
  IntermediateHitDoublets(const IntermediateHitDoublets& rh); // only to make ROOT dictionary generation happy
  ~IntermediateHitDoublets() = default;

  void swap(IntermediateHitDoublets& rh) {
    std::swap(seedingLayers_, rh.seedingLayers_);
    std::swap(regions_, rh.regions_);
    std::swap(layerPairs_, rh.layerPairs_);
  }

  void reserve(size_t nregions, size_t nlayersets) {
    regions_.reserve(nregions);
    layerPairs_.reserve(nregions*nlayersets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    layerPairs_.shrink_to_fit();
  }

  void beginRegion(const TrackingRegion *region) {
    regions_.emplace_back(region, layerPairs_.size());
  }

  void addDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets, LayerHitMapCache&& cache) {
    layerPairs_.emplace_back(layerSet, std::move(doublets), std::move(cache));
    regions_.back().setLayerSetsEnd(layerPairs_.size());
  }

  const SeedingLayerSetsHits& seedingLayerHits() const { return *seedingLayers_; }
  bool empty() const { return regions_.empty(); }
  size_t regionSize() const { return regions_.size(); }
  size_t layerPairsSize() const { return layerPairs_.size(); }

  const_iterator begin() const { return const_iterator(this, regions_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, regions_.end()); }
  const_iterator cend() const { return end(); }

  // used internally
  std::vector<RegionIndex>::const_iterator regionsBegin() const { return regions_.begin(); }
  std::vector<RegionIndex>::const_iterator regionsEnd() const { return regions_.end(); }
  std::vector<LayerPairHitDoublets>::const_iterator layerSetsBegin() const { return layerPairs_.begin(); }
  std::vector<LayerPairHitDoublets>::const_iterator layerSetsEnd() const { return layerPairs_.end(); }

private:
  const SeedingLayerSetsHits *seedingLayers_;

  std::vector<RegionIndex> regions_;
  std::vector<LayerPairHitDoublets> layerPairs_;
};

#endif

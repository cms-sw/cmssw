#ifndef RecoTracker_TkHitPairs_IntermediateHitDoublets_h
#define RecoTracker_TkHitPairs_IntermediateHitDoublets_h

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

namespace ihd {
  /**
   * Class to hold TrackingRegion and begin+end indices to a vector of
   * seeding layer sets.
   *
   * The LayerHitMapCache is also hold here as it is a per-region object.
   *
   * In practice the vector being indexed can be anything.
   */
  class RegionIndex {
  public:
    RegionIndex(const TrackingRegion* reg, unsigned int ind)
        : region_(reg), layerSetBeginIndex_(ind), layerSetEndIndex_(ind) {}
    RegionIndex(RegionIndex&&) = default;
    RegionIndex& operator=(RegionIndex&&) = default;

    void setLayerSetsEnd(unsigned int end) { layerSetEndIndex_ = end; }

    const TrackingRegion& region() const { return *region_; }

    LayerHitMapCache& layerHitMapCache() { return cache_; }
    const LayerHitMapCache& layerHitMapCache() const { return cache_; }

    unsigned int layerSetBeginIndex() const { return layerSetBeginIndex_; }
    unsigned int layerSetEndIndex() const { return layerSetEndIndex_; }

  private:
    const TrackingRegion* region_;  /// pointer to TrackingRegion (owned elsewhere)
    LayerHitMapCache cache_;
    unsigned int layerSetBeginIndex_;  /// index of the beginning of layer sets of this region
    unsigned int layerSetEndIndex_;    /// index of the end (one-past-last) of layer sets of this region
  };

  /**
   * Helper class to provide nice interface to loop over the layer sets of a region
   *
   * \tparam T Concrete type in a vector<T> actually holding the layer sets
   *
   * Templatized because used here and in RegionSeedingHitSets
   */
  template <typename T>
  class RegionLayerSets {
  public:
    using const_iterator = typename std::vector<T>::const_iterator;

    // Taking T* to have compatible interface with IntermediateHitTriplets::RegionLayerSets
    template <typename TMP>
    RegionLayerSets(const TrackingRegion* region,
                    const LayerHitMapCache* cache,
                    const TMP*,
                    const_iterator begin,
                    const_iterator end)
        : region_(region), cache_(cache), layerSetsBegin_(begin), layerSetsEnd_(end) {}

    const TrackingRegion& region() const { return *region_; }
    const LayerHitMapCache& layerHitMapCache() const { return *cache_; }

    const_iterator begin() const { return layerSetsBegin_; }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return layerSetsEnd_; }
    const_iterator cend() const { return end(); }

  private:
    const TrackingRegion* region_;
    const LayerHitMapCache* cache_;
    const const_iterator layerSetsBegin_;
    const const_iterator layerSetsEnd_;
  };

  /**
   * Helper class providing a generic iterator to loop over
   * TrackingRegions of IntermediateHitDoublets,
   * IntermediateHitTriplets, or RegionsSeedingHitSets
   *
   * \tparam ValueType   Type to be returned by operator*() (should be something inexpensive)
   * \tparam HitSetType  Type of the holder of data (currently IntermediateHitDoublets, IntermediateHitTriplets, or RegionsSeedingHitSets)
   */
  template <typename ValueType, typename HitSetType>
  class const_iterator {
  public:
    using internal_iterator_type = typename std::vector<RegionIndex>::const_iterator;
    using value_type = ValueType;
    using difference_type = internal_iterator_type::difference_type;

    const_iterator(const HitSetType* hst, internal_iterator_type iter) : hitSets_(hst), iter_(iter) {}

    value_type operator*() const {
      return value_type(&(iter_->region()),
                        &(iter_->layerHitMapCache()),
                        hitSets_,
                        hitSets_->layerSetsBegin() + iter_->layerSetBeginIndex(),
                        hitSets_->layerSetsBegin() + iter_->layerSetEndIndex());
    }

    const_iterator& operator++() {
      ++iter_;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator clone(*this);
      ++iter_;
      return clone;
    }

    bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }

  private:
    const HitSetType* hitSets_;
    internal_iterator_type iter_;
  };
}  // namespace ihd

/**
 * Container of temporary information delivered from hit pair
 * generator to hit triplet generator via edm::Event.
 *
 * The iterator loops over regions, and the value_type of that has an
 * iterator looping over the layer pairs of the region.
 *
 * Pointers to SeedingLayerSetsHits and TrackingRegion are stored, so
 * the lifetime of those objects should be at least as long as the
 * lifetime of this object.
 */
class IntermediateHitDoublets {
public:
  using LayerPair = std::tuple<SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex>;
  using RegionIndex = ihd::RegionIndex;

  /**
   * This class stores the indices of a layer pair, and the doublets
   * generated from there.
   *
   * The layer indices are those from SeedingLayerSetsHits.
   *
   * Since the doublets are stored directly here, the same class works
   * nicely for both storage and use.
   */
  class LayerPairHitDoublets {
  public:
    LayerPairHitDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets)
        : layerPair_(layerSet[0].index(), layerSet[1].index()), doublets_(std::move(doublets)) {}

    const LayerPair& layerPair() const { return layerPair_; }
    SeedingLayerSetsHits::LayerIndex innerLayerIndex() const { return std::get<0>(layerPair_); }
    SeedingLayerSetsHits::LayerIndex outerLayerIndex() const { return std::get<1>(layerPair_); }

    const HitDoublets& doublets() const { return doublets_; }

  private:
    LayerPair layerPair_;   /// pair of indices to the layer
    HitDoublets doublets_;  /// container of the doublets
  };

  ////////////////////

  /// Helper class providing nice interface to loop over layer sets of a region
  using RegionLayerSets = ihd::RegionLayerSets<LayerPairHitDoublets>;

  ////////////////////

  /// Iterator over regions
  using const_iterator = ihd::const_iterator<RegionLayerSets, IntermediateHitDoublets>;

  ////////////////////

  /// Helper class enforcing correct way of filling the doublets of a region
  class RegionFiller {
  public:
    RegionFiller() : obj_(nullptr) {}
    explicit RegionFiller(IntermediateHitDoublets* obj) : obj_(obj) {}

    ~RegionFiller() = default;

    bool valid() const { return obj_ != nullptr; }

    LayerHitMapCache& layerHitMapCache() { return obj_->regions_.back().layerHitMapCache(); }

    void addDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets) {
      obj_->layerPairs_.emplace_back(layerSet, std::move(doublets));
      obj_->regions_.back().setLayerSetsEnd(obj_->layerPairs_.size());
    }

  private:
    IntermediateHitDoublets* obj_;
  };

  // allows declaring local variables with auto
  static RegionFiller dummyFiller() { return RegionFiller(); }

  ////////////////////

  IntermediateHitDoublets() : seedingLayers_(nullptr) {}
  explicit IntermediateHitDoublets(const SeedingLayerSetsHits* seedingLayers) : seedingLayers_(seedingLayers) {}
  IntermediateHitDoublets(const IntermediateHitDoublets& rh);  // only to make ROOT dictionary generation happy
  IntermediateHitDoublets(IntermediateHitDoublets&&) = default;
  IntermediateHitDoublets& operator=(IntermediateHitDoublets&&) = default;
  ~IntermediateHitDoublets() = default;

  void reserve(size_t nregions, size_t nlayersets) {
    regions_.reserve(nregions);
    layerPairs_.reserve(nregions * nlayersets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    layerPairs_.shrink_to_fit();
  }

  RegionFiller beginRegion(const TrackingRegion* region) {
    regions_.emplace_back(region, layerPairs_.size());
    return RegionFiller(this);
  }

  const SeedingLayerSetsHits& seedingLayerHits() const { return *seedingLayers_; }
  bool empty() const { return regions_.empty(); }
  size_t regionSize() const { return regions_.size(); }
  size_t layerPairsSize() const { return layerPairs_.size(); }

  const_iterator begin() const { return const_iterator(this, regions_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, regions_.end()); }
  const_iterator cend() const { return end(); }

  // used internally by all the helper classes
  std::vector<RegionIndex>::const_iterator regionsBegin() const { return regions_.begin(); }
  std::vector<RegionIndex>::const_iterator regionsEnd() const { return regions_.end(); }
  std::vector<LayerPairHitDoublets>::const_iterator layerSetsBegin() const { return layerPairs_.begin(); }
  std::vector<LayerPairHitDoublets>::const_iterator layerSetsEnd() const { return layerPairs_.end(); }

private:
  const SeedingLayerSetsHits* seedingLayers_;  /// Pointer to SeedingLayerSetsHits (owned elsewhere)

  std::vector<RegionIndex> regions_;  /// Container of regions, each element has indices pointing to layerPairs_
  std::vector<LayerPairHitDoublets> layerPairs_;  /// Container of layer pairs and doublets for all regions
};

#endif

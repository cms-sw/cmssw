#ifndef RecoTracker_PixelSeeding_IntermediateHitTriplets_h
#define RecoTracker_PixelSeeding_IntermediateHitTriplets_h

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplets.h"

/**
 * Container of temporary information delivered from hit triplet
 * generator to hit quadruplet generator via edm::Event.
 *
 * The iterator loops over regions, the value_type of that has an
 * iterator looping over the layer triplets of the region, and the
 * value_type of that has an iterator looping over the hit triplets of
 * the layer triplet.
 *
 * Pointers to SeedingLayerSetsHits and TrackingRegion are stored, so
 * the lifetime of those objects should be at least as long as the
 * lifetime of this object.
 */
class IntermediateHitTriplets {
public:
  using LayerPair = std::tuple<SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex>;
  using LayerTriplet =
      std::tuple<SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex>;
  using RegionIndex = ihd::RegionIndex;

  ////////////////////

  /**
   * Helper class holding the layer triplet indices (to
   * SeedingLayerSetsHits), and indices of the hit triplets from this
   * layer triplet (to the hit triplet vector)
   *
   * As only the indices of hit triplets are stored, a separate class
   * (LayerHitTriplets) is provided with nicer interface.
   */
  class PLayerHitTriplets {
  public:
    PLayerHitTriplets(const LayerTriplet &layerTriplet, unsigned int tripletsBegin)
        : layerTriplet_(layerTriplet), tripletsBegin_(tripletsBegin), tripletsEnd_(tripletsBegin) {}

    void setTripletsEnd(unsigned int end) { tripletsEnd_ = end; }

    const LayerTriplet &layerTriplet() const { return layerTriplet_; }

    unsigned int tripletsBegin() const { return tripletsBegin_; }
    unsigned int tripletsEnd() const { return tripletsEnd_; }

  private:
    LayerTriplet layerTriplet_;
    unsigned int tripletsBegin_;
    unsigned int tripletsEnd_;
  };

  ////////////////////

  /**
   * Helper class providing a nice interface for the hit triplets of a
   * layer triplet.
   */
  class LayerHitTriplets {
  public:
    using const_iterator = std::vector<OrderedHitTriplet>::const_iterator;

    LayerHitTriplets(const IntermediateHitTriplets *hitSets, const PLayerHitTriplets *layerTriplet)
        : hitSets_(hitSets), layerTriplet_(layerTriplet) {}

    using TripletRange =
        std::pair<std::vector<OrderedHitTriplet>::const_iterator, std::vector<OrderedHitTriplet>::const_iterator>;

    SeedingLayerSetsHits::LayerIndex innerLayerIndex() const { return std::get<0>(layerTriplet_->layerTriplet()); }
    SeedingLayerSetsHits::LayerIndex middleLayerIndex() const { return std::get<1>(layerTriplet_->layerTriplet()); }
    SeedingLayerSetsHits::LayerIndex outerLayerIndex() const { return std::get<2>(layerTriplet_->layerTriplet()); }

    const_iterator begin() const { return hitSets_->tripletsBegin() + layerTriplet_->tripletsBegin(); }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return hitSets_->tripletsBegin() + layerTriplet_->tripletsEnd(); }
    const_iterator cend() const { return end(); }

  private:
    const IntermediateHitTriplets *hitSets_;
    const PLayerHitTriplets *layerTriplet_;
  };

  ////////////////////

  /**
   * Helper class to provide nice interface to loop over the layer sets of a region
   *
   * The value_type of the iterator is LayerHitTriplets, which has an
   * iterator for the hit triplets.
   *
   * Can not use ihd::RegionLayerSets<T> here because of having
   * separate classes for storage (PLayerHitTriplets) and use
   * (LayerHitTriplets).
   */
  class RegionLayerSets {
  public:
    using PLayerHitTripletsConstIterator = std::vector<PLayerHitTriplets>::const_iterator;
    using TripletConstIterator = std::vector<OrderedHitTriplet>::const_iterator;

    class const_iterator {
    public:
      using internal_iterator_type = PLayerHitTripletsConstIterator;
      using value_type = LayerHitTriplets;
      using difference_type = internal_iterator_type::difference_type;

      struct end_tag {};

      /**
       * Constructor for an iterator pointing to a valid element
       */
      const_iterator(const IntermediateHitTriplets *hitSets, const RegionLayerSets *regionLayerSets)
          : hitSets_(hitSets), regionLayerSets_(regionLayerSets), iter_(regionLayerSets->layerSetsBegin()) {
        assert(regionLayerSets->layerSetsBegin() != regionLayerSets->layerSetsEnd());
      }

      /**
       * Constructor for an iterator pointing to an invalid element (i.e. end)
       *
       * The end_tag parameter is used to differentiate this constructor from the other one.
       */
      const_iterator(const IntermediateHitTriplets *hitSets, const RegionLayerSets *regionLayerSets, end_tag)
          : iter_(regionLayerSets->layerSetsEnd()) {}

      value_type operator*() const { return value_type(hitSets_, &(*iter_)); }

      const_iterator &operator++() {
        ++iter_;
        return *this;
      }

      const_iterator operator++(int) {
        const_iterator clone(*this);
        operator++();
        return clone;
      }

      bool operator==(const const_iterator &other) const { return iter_ == other.iter_; }
      bool operator!=(const const_iterator &other) const { return !operator==(other); }

    private:
      const IntermediateHitTriplets *hitSets_;
      const RegionLayerSets *regionLayerSets_;
      internal_iterator_type iter_;
    };

    RegionLayerSets(const TrackingRegion *region,
                    const LayerHitMapCache *cache,
                    const IntermediateHitTriplets *hitSets,
                    PLayerHitTripletsConstIterator tripletBegin,
                    PLayerHitTripletsConstIterator tripletEnd)
        : region_(region), cache_(cache), hitSets_(hitSets), layerSetsBegin_(tripletBegin), layerSetsEnd_(tripletEnd) {}

    const TrackingRegion &region() const { return *region_; }
    const LayerHitMapCache &layerHitMapCache() const { return *cache_; }
    size_t layerTripletsSize() const { return std::distance(layerSetsBegin_, layerSetsEnd_); }

    const_iterator begin() const {
      if (layerSetsBegin_ != layerSetsEnd_)
        return const_iterator(hitSets_, this);
      else
        return end();
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return const_iterator(hitSets_, this, const_iterator::end_tag()); }
    const_iterator cend() const { return end(); }

    // used internally by the LayerHitTriplets helper class
    PLayerHitTripletsConstIterator layerSetsBegin() const { return layerSetsBegin_; }
    PLayerHitTripletsConstIterator layerSetsEnd() const { return layerSetsEnd_; }

  private:
    const TrackingRegion *region_ = nullptr;
    const LayerHitMapCache *cache_ = nullptr;
    const IntermediateHitTriplets *hitSets_ = nullptr;
    const PLayerHitTripletsConstIterator layerSetsBegin_;
    const PLayerHitTripletsConstIterator layerSetsEnd_;
  };

  ////////////////////

  /// Iterator over regions
  using const_iterator = ihd::const_iterator<RegionLayerSets, IntermediateHitTriplets>;

  ////////////////////

  /// Helper class enforcing correct way of filling the doublets of a region
  class RegionFiller {
  public:
    RegionFiller() : obj_(nullptr) {}
    explicit RegionFiller(IntermediateHitTriplets *obj) : obj_(obj) {}

    ~RegionFiller() = default;

    bool valid() const { return obj_ != nullptr; }

    LayerHitMapCache &layerHitMapCache() { return obj_->regions_.back().layerHitMapCache(); }

    void addTriplets(const LayerPair &layerPair,
                     const std::vector<SeedingLayerSetsHits::SeedingLayer> &thirdLayers,
                     const OrderedHitTriplets &triplets,
                     const std::vector<int> &thirdLayerIndex,
                     const std::vector<size_t> &permutations) {
      assert(triplets.size() == thirdLayerIndex.size());
      assert(triplets.size() == permutations.size());

      if (triplets.empty()) {
        return;
      }

      int prevLayer = -1;
      for (size_t i = 0, size = permutations.size(); i < size; ++i) {
        // We go through the 'triplets' in the order defined by
        // 'permutations', which is sorted such that we first go through
        // triplets from (3rd) layer 0, then layer 1 and so on.
        const size_t realIndex = permutations[i];

        const int layer = thirdLayerIndex[realIndex];
        if (layer != prevLayer) {
          prevLayer = layer;
          obj_->layerTriplets_.emplace_back(
              LayerTriplet(std::get<0>(layerPair), std::get<1>(layerPair), thirdLayers[layer].index()),
              obj_->hitTriplets_.size());
        }

        obj_->hitTriplets_.emplace_back(triplets[realIndex]);
        obj_->layerTriplets_.back().setTripletsEnd(obj_->hitTriplets_.size());
      }

      obj_->regions_.back().setLayerSetsEnd(obj_->layerTriplets_.size());
    }

  private:
    IntermediateHitTriplets *obj_;
  };

  // allows declaring local variables with auto
  static RegionFiller dummyFiller() { return RegionFiller(); }

  ////////////////////

  IntermediateHitTriplets() : seedingLayers_(nullptr) {}
  explicit IntermediateHitTriplets(const SeedingLayerSetsHits *seedingLayers) : seedingLayers_(seedingLayers) {}
  IntermediateHitTriplets(const IntermediateHitTriplets &rh);  // only to make ROOT dictionary generation happy
  IntermediateHitTriplets(IntermediateHitTriplets &&) = default;
  IntermediateHitTriplets &operator=(IntermediateHitTriplets &&) = default;
  ~IntermediateHitTriplets() = default;

  void reserve(size_t nregions, size_t nlayersets, size_t ntriplets) {
    regions_.reserve(nregions);
    layerTriplets_.reserve(nregions * nlayersets);
    hitTriplets_.reserve(ntriplets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    layerTriplets_.shrink_to_fit();
    hitTriplets_.shrink_to_fit();
  }

  RegionFiller beginRegion(const TrackingRegion *region) {
    regions_.emplace_back(region, layerTriplets_.size());
    return RegionFiller(this);
  }

  const SeedingLayerSetsHits &seedingLayerHits() const { return *seedingLayers_; }
  bool empty() const { return regions_.empty(); }
  size_t regionSize() const { return regions_.size(); }
  size_t tripletsSize() const { return hitTriplets_.size(); }

  const_iterator begin() const { return const_iterator(this, regions_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, regions_.end()); }
  const_iterator cend() const { return end(); }

  // used internally by all the helper classes
  std::vector<RegionIndex>::const_iterator regionsBegin() const { return regions_.begin(); }
  std::vector<RegionIndex>::const_iterator regionsEnd() const { return regions_.end(); }
  std::vector<PLayerHitTriplets>::const_iterator layerSetsBegin() const { return layerTriplets_.begin(); }
  std::vector<PLayerHitTriplets>::const_iterator layerSetsEnd() const { return layerTriplets_.end(); }
  std::vector<OrderedHitTriplet>::const_iterator tripletsBegin() const { return hitTriplets_.begin(); }
  std::vector<OrderedHitTriplet>::const_iterator tripletsEnd() const { return hitTriplets_.end(); }

private:
  const SeedingLayerSetsHits *seedingLayers_;  /// Pointer to SeedingLayerSetsHits (owned elsewhere)

  std::vector<RegionIndex> regions_;  /// Container of regions, each element has indices pointing to layerTriplets_
  std::vector<PLayerHitTriplets>
      layerTriplets_;  /// Container of layer triplets, each element has indices pointing to hitTriplets_
  std::vector<OrderedHitTriplet> hitTriplets_;  /// Container of hit triplets for all layer triplets and regions
};

#endif

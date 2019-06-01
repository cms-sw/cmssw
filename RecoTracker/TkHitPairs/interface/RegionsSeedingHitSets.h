#ifndef RecoTracker_TkHitPairs_RegionsSeedingHitSets_H
#define RecoTracker_TkHitPairs_RegionsSeedingHitSets_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

/**
 * Class to store SeedingHitSets (doublet/triplet/quadruplet) per TrackingRegion
 *
 * Defined in this package instead of RecoTracker/TkSeedingLayers to avoid circular dependencies
 */
class RegionsSeedingHitSets {
public:
  /// Helper class containing a region and indices to hitSets_
  using RegionIndex = ihd::RegionIndex;

  /// Helper class providing nice interface to loop over hit sets of a region
  using RegionSeedingHitSets = ihd::RegionLayerSets<SeedingHitSet>;

  /// Iterator over regions
  using const_iterator = ihd::const_iterator<RegionSeedingHitSets, RegionsSeedingHitSets>;

  /// Helper class enforcing correct way of filling the doublets of a region
  class RegionFiller {
  public:
    RegionFiller() : obj_(nullptr) {}
    explicit RegionFiller(RegionsSeedingHitSets* obj) : obj_(obj) {}

    ~RegionFiller() {
      if (obj_)
        obj_->regions_.back().setLayerSetsEnd(obj_->hitSets_.size());
    }

    bool valid() const { return obj_ != nullptr; }

    template <typename... Args>
    void emplace_back(Args&&... args) {
      obj_->hitSets_.emplace_back(std::forward<Args>(args)...);
    }

  private:
    RegionsSeedingHitSets* obj_;
  };

  // allows declaring local variables with auto
  static RegionFiller dummyFiller() { return RegionFiller(); }

  // constructors
  RegionsSeedingHitSets() = default;
  RegionsSeedingHitSets(const RegionsSeedingHitSets&) = delete;
  RegionsSeedingHitSets& operator=(const RegionsSeedingHitSets&) = delete;
  RegionsSeedingHitSets(RegionsSeedingHitSets&&) = default;
  RegionsSeedingHitSets& operator=(RegionsSeedingHitSets&&) = default;
  ~RegionsSeedingHitSets() = default;

  void reserve(size_t nregions, size_t nhitsets) {
    regions_.reserve(nregions);
    hitSets_.reserve(nhitsets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    hitSets_.shrink_to_fit();
  }

  RegionFiller beginRegion(const TrackingRegion* region) {
    regions_.emplace_back(region, hitSets_.size());
    return RegionFiller(this);
  }

  bool empty() const { return regions_.empty(); }
  size_t regionSize() const { return regions_.size(); }
  size_t size() const { return hitSets_.size(); }

  const_iterator begin() const { return const_iterator(this, regions_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, regions_.end()); }
  const_iterator cend() const { return end(); }

  // used internally by the helper classes
  std::vector<SeedingHitSet>::const_iterator layerSetsBegin() const { return hitSets_.begin(); }
  std::vector<SeedingHitSet>::const_iterator layerSetsEnd() const { return hitSets_.end(); }

private:
  std::vector<RegionIndex> regions_;    /// Container of regions, each element has indices pointing to hitSets_
  std::vector<SeedingHitSet> hitSets_;  /// Container of hit sets for all regions
};

#endif

#ifndef RecoTracker_TkHitPairs_RegionsSeedingHitSets_H
#define RecoTracker_TkHitPairs_RegionsSeedingHitSets_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

// defined in this package instead of RecoTracker/TkSeedingLayers to avoid circular dependencies

class RegionsSeedingHitSets {
public:
  using RegionIndex = ihd::RegionIndex;

  using RegionSeedingHitSets = ihd::RegionLayerHits<SeedingHitSet>;
  using const_iterator = ihd::const_iterator<RegionSeedingHitSets, RegionsSeedingHitSets>;

  // helper class to enforce correct usage
  class RegionFiller {
  public:
    RegionFiller(): obj_(nullptr) {}
    explicit RegionFiller(RegionsSeedingHitSets* obj): obj_(obj) {}

    ~RegionFiller() {
      if(obj_) obj_->regions_.back().setLayerSetsEnd(obj_->hitSets_.size());
    }

    bool valid() const { return obj_ != nullptr; }

    template <typename... Args>
    void emplace_back(Args&&... args) {
      obj_->hitSets_.emplace_back(std::forward<Args>(args)...);
    }
  private:
    RegionsSeedingHitSets *obj_;
  };

  static RegionFiller dummyFiller() { return RegionFiller(); }

  // constructors
  RegionsSeedingHitSets() = default;
  ~RegionsSeedingHitSets() = default;

  void swap(RegionsSeedingHitSets& rh) {
    regions_.swap(rh.regions_);
    hitSets_.swap(rh.hitSets_);
  }

  void reserve(size_t nregions, size_t nhitsets) {
    regions_.reserve(nregions);
    hitSets_.reserve(nhitsets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    hitSets_.shrink_to_fit();
  }

  RegionFiller beginRegion(const TrackingRegion *region) {
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

  // Used internally
  std::vector<SeedingHitSet>::const_iterator layerSetsBegin() const { return hitSets_.begin(); }
  std::vector<SeedingHitSet>::const_iterator layerSetsEnd() const { return hitSets_.end(); }

private:
  std::vector<RegionIndex> regions_;
  std::vector<SeedingHitSet> hitSets_;
};

#endif

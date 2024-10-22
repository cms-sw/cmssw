#ifndef RecoTracker_PixelSeeding_OrderedHitSeeds_h
#define RecoTracker_PixelSeeding_OrderedHitSeeds_h

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitSeeds : public std::vector<SeedingHitSet>, public OrderedSeedingHits {
public:
  ~OrderedHitSeeds() override {}

  unsigned int size() const override { return std::vector<SeedingHitSet>::size(); }

  const SeedingHitSet& operator[](unsigned int i) const override { return std::vector<SeedingHitSet>::operator[](i); }
};
#endif

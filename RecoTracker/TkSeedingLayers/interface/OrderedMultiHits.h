#ifndef OrderedMultiHits_H
#define OrderedMultiHits_H

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"

class OrderedMultiHits : public std::vector<SeedingHitSet>, public OrderedSeedingHits {
public:

  ~OrderedMultiHits() override{}

  unsigned int size() const override { return std::vector<SeedingHitSet>::size(); }

  const SeedingHitSet & operator[](unsigned int i) const override {
    return std::vector<SeedingHitSet>::operator[](i);
  }

};
#endif

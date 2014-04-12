#ifndef OrderedMultiHits_H
#define OrderedMultiHits_H

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"

class OrderedMultiHits : public std::vector<SeedingHitSet>, public OrderedSeedingHits {
public:

  virtual ~OrderedMultiHits(){}

  virtual unsigned int size() const { return std::vector<SeedingHitSet>::size(); }

  virtual const SeedingHitSet & operator[](unsigned int i) const {
    return std::vector<SeedingHitSet>::operator[](i);
  }

};
#endif

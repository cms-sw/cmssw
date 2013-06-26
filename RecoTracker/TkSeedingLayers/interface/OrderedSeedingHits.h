#ifndef TkSeedingLayers_OrderedSeedingHits_H
#define TkSeedingLayers_OrderedSeedingHits_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include <vector>

class OrderedSeedingHits {
public:

  virtual ~OrderedSeedingHits(){}
  virtual unsigned int size() const = 0;
  virtual const SeedingHitSet & operator[](unsigned int i) const = 0;

};

#endif

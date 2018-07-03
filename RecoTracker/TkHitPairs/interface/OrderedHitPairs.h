#ifndef OrderedHitPairs_H
#define OrderedHitPairs_H

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitPairs : public std::vector<OrderedHitPair>, public OrderedSeedingHits {
public:

  ~OrderedHitPairs() override{}

  unsigned int size() const override { return std::vector<OrderedHitPair>::size(); }

  const OrderedHitPair& operator[](unsigned int i) const override { 
    return std::vector<OrderedHitPair>::operator[](i); 
  }

};
#endif

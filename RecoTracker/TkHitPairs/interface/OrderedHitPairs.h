#ifndef OrderedHitPairs_H
#define OrderedHitPairs_H

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitPairs : public std::vector<OrderedHitPair>, public OrderedSeedingHits {
public:

  virtual ~OrderedHitPairs(){}

  virtual unsigned int size() const { return std::vector<OrderedHitPair>::size(); }

  virtual const OrderedHitPair& operator[](unsigned int i) const { 
    return std::vector<OrderedHitPair>::operator[](i); 
  }

};
#endif

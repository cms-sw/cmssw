#ifndef RecoTracker_PixelSeeding_OrderedHitTriplets_h
#define RecoTracker_PixelSeeding_OrderedHitTriplets_h

#include <vector>
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitTriplets : public std::vector<OrderedHitTriplet>, public OrderedSeedingHits {
public:
  ~OrderedHitTriplets() override {}

  unsigned int size() const override { return std::vector<OrderedHitTriplet>::size(); }

  const OrderedHitTriplet& operator[](unsigned int i) const override {
    return std::vector<OrderedHitTriplet>::operator[](i);
  }
};
#endif

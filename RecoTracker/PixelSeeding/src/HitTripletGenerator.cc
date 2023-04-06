#include "RecoTracker/PixelSeeding/interface/HitTripletGenerator.h"

HitTripletGenerator::HitTripletGenerator(unsigned int nSize) : localRA(nSize) {}

const OrderedHitTriplets& HitTripletGenerator::run(const TrackingRegion& region,
                                                   const edm::Event& ev,
                                                   const edm::EventSetup& es) {
  assert(theTriplets.empty());
  assert(theTriplets.capacity() == 0);
  theTriplets.reserve(localRA.upper());
  hitTriplets(region, theTriplets, ev, es);
  localRA.update(theTriplets.size());
  theTriplets.shrink_to_fit();
  return theTriplets;
}

void HitTripletGenerator::clear() {
  theTriplets.clear();
  theTriplets.shrink_to_fit();
}

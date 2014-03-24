#ifndef TkTrackingRegions_OrderedHitsGenerator_H
#define TkTrackingRegions_OrderedHitsGenerator_H

#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class TrackingRegion;
namespace edm { class Event; class EventSetup; class ConsumesCollector;}

class OrderedHitsGenerator {
public:
  OrderedHitsGenerator() : theMaxElement(0){}
  virtual ~OrderedHitsGenerator() {}

  virtual const OrderedSeedingHits & run( 
      const TrackingRegion& reg, const edm::Event & ev, const edm::EventSetup& es ) = 0;

  virtual void clear() { }  //fixme: should be purely virtual!

  unsigned int theMaxElement;
};

#endif

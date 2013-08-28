#ifndef MultiHitGenerator_H
#define MultiHitGenerator_H

/** abstract interface for generators of hit triplets pairs
 *  compatible with a TrackingRegion.
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedMultiHits.h"

class TrackingRegion;
namespace edm { class Event; class EventSetup; }
#include <vector>

class MultiHitGenerator : public OrderedHitsGenerator {
public:

  MultiHitGenerator(unsigned int size=500);

  virtual ~MultiHitGenerator() { }

  virtual const OrderedMultiHits & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);

  // temporary interface, for bckwd compatibility
  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & prs,
       const edm::EventSetup& es){}

  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & prs,
      const edm::Event & ev,  const edm::EventSetup& es) = 0;

  virtual void clear();

private:
  OrderedMultiHits theHitSets;

};


#endif

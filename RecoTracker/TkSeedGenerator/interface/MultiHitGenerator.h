#ifndef MultiHitGenerator_H
#define MultiHitGenerator_H

/** abstract interface for generators of hit triplets pairs
 *  compatible with a TrackingRegion.
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedMultiHits.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"


class TrackingRegion;
namespace edm { class Event; class EventSetup; }
#include <vector>

class MultiHitGenerator : public OrderedHitsGenerator {
public:

  MultiHitGenerator(unsigned int size=500);

  virtual ~MultiHitGenerator() { }

  virtual const OrderedMultiHits & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es) final;

  // temporary interface, for bckwd compatibility
  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & prs,
       const edm::EventSetup& es){}

  virtual void hitSets( const TrackingRegion& reg, OrderedMultiHits & prs,
      const edm::Event & ev,  const edm::EventSetup& es) = 0;

  virtual void clear();

private:
  OrderedMultiHits theHitSets;

protected:
  using cacheHitPointer = std::unique_ptr<BaseTrackerRecHit>;
  using cacheHits=std::vector<cacheHitPointer>;
  cacheHits cache; // ownes what is by reference above...

};


#endif

#ifndef HitPairGenerator_H
#define HitPairGenerator_H

#include <vector>

/** abstract interface for generators of ordered RecHit pairs
 *  compatible with a TrackingRegion. 
 *  This is used by the HitPairSeedGenerator to produce TrajectorySeeds
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"

class TrackingRegion;
namespace edm { class Event; class EventSetup; }

class HitPairGenerator : public OrderedHitsGenerator {
public:

  HitPairGenerator(unsigned int size=30000);

  virtual ~HitPairGenerator() { }

  virtual const OrderedHitPairs & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);

  // temporary interface for backward compatibility only
  virtual void hitPairs( 
    const TrackingRegion& reg, OrderedHitPairs & prs, const edm::EventSetup& es) {}

  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs, 
      const edm::Event & ev,  const edm::EventSetup& es) = 0;

  virtual HitPairGenerator* clone() const = 0;

  virtual void clear() { thePairs.clear(); } 

private:
  OrderedHitPairs thePairs;

};

#endif

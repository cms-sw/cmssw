#ifndef HitPairGenerator_H
#define HitPairGenerator_H

#include <vector>

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"

/** abstract interface for generators of ordered RecHit pairs
 *  compatible with a TrackingRegion. 
 *  This is used by the HitPairSeedGenerator to produce TrajectorySeeds
 */

class TrackingRegion;

class HitPairGenerator {
public:

  virtual ~HitPairGenerator() { }

  virtual OrderedHitPairs hitPairs( const TrackingRegion& region ) {
    OrderedHitPairs pairs; 
    hitPairs(region, pairs);
    return pairs;
  } 
  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs) = 0;

  virtual HitPairGenerator* clone() const = 0;

};

#endif

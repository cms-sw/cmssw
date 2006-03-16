#ifndef HitPairGenerator_H
#define HitPairGenerator_H

#include <vector>

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
/** abstract interface for generators of ordered RecHit pairs
 *  compatible with a TrackingRegion. 
 *  This is used by the HitPairSeedGenerator to produce TrajectorySeeds
 */

class TrackingRegion;

class HitPairGenerator {
public:

  virtual ~HitPairGenerator() { }

  virtual OrderedHitPairs hitPairs( const TrackingRegion& region,const edm::EventSetup& iSetup ) {
    OrderedHitPairs pairs; 
    hitPairs(region, pairs, iSetup);
    return pairs;
  } 
  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs,const edm::EventSetup& iSetup) = 0;

  virtual HitPairGenerator* clone() const = 0;

};

#endif

#ifndef HitPairGenerator_H
#define HitPairGenerator_H

#include <vector>

/** abstract interface for generators of ordered RecHit pairs
 *  compatible with a TrackingRegion. 
 *  This is used by the HitPairSeedGenerator to produce TrajectorySeeds
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

class TrackingRegion;
namespace edm { class Event; class EventSetup; }

class HitPairGenerator : public OrderedHitsGenerator {
public:

  explicit HitPairGenerator(unsigned int size=7500);
  HitPairGenerator(HitPairGenerator const & other) : localRA(other.localRA.mean()){}

  virtual ~HitPairGenerator() { }

  virtual const OrderedHitPairs & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);

  // temporary interface for backward compatibility only
  virtual void hitPairs( 
    const TrackingRegion& reg, OrderedHitPairs & prs, const edm::EventSetup& es) {}

  // new interface with no temphits copy
  virtual HitDoublets doublets( const TrackingRegion& reg, 
			     const edm::Event & ev,  const edm::EventSetup& es) {
    assert(0=="not implemented");
  }


  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs, 
      const edm::Event & ev,  const edm::EventSetup& es) = 0;

  virtual HitPairGenerator* clone() const = 0;

  virtual void clear() final;

private:
  OrderedHitPairs thePairs;
  edm::RunningAverage localRA;

};

#endif

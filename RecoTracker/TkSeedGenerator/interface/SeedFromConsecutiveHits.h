#ifndef SeedFromConsecutiveHits_H
#define SeedFromConsecutiveHits_H

/** \class SeedFromConsecutiveHits
 * OBSOLETE !!!! USER SeedFromConsecutiveHitsCreator
 */

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

class SeedFromConsecutiveHits{

public:
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
 
  SeedFromConsecutiveHits(const SeedingHitSet & hits,
    const GlobalPoint& vertexPos, const GlobalError& vertexErr,
    const edm::EventSetup& es, float ptMin, double theBOFFMomentum=-1.0) 
  {
    GlobalTrackingRegion region( ptMin, vertexPos, sqrt(vertexErr.cxx()), sqrt(vertexErr.czz()) );  
    SeedFromConsecutiveHitsCreator creator("PropagatorWithMaterial",theBOFFMomentum);
    theSeed.clear();
    creator.trajectorySeed(theSeed, hits, region, es); 
  } 
 
  virtual  ~SeedFromConsecutiveHits(){};

  bool isValid() {return  theSeed.size()>0 ; }
 
  TrajectorySeed TrajSeed(){ return theSeed.back(); }

private:
  TrajectorySeedCollection theSeed;
};

#endif

#ifndef SeedFromConsecutiveHits_H
#define SeedFromConsecutiveHits_H

/** \class SeedFromConsecutiveHits
 * Seed constructed from the outer and inner RecHit (order important)
 * and the vertex constraints
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


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

class DetLayer;
class SeedFromConsecutiveHits{
 public:
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  
  // obsolete!
  SeedFromConsecutiveHits( const TrackingRecHit* outerHit,
			   const TrackingRecHit* innerHit,
			   const GlobalPoint& vertexPos,
			   const GlobalError& vertexErr,
			   const edm::EventSetup& iSetup,
			   const edm::ParameterSet& p
			   );

  SeedFromConsecutiveHits(const SeedingHitSet & hits,
    const GlobalPoint& vertexPos,
    const GlobalError& vertexErr,
    const edm::EventSetup& es);
  
  virtual  ~SeedFromConsecutiveHits(){};

  bool isValid() {return isValid_;}
  
  PropagationDirection direction(){
    //as in ORCA
    return alongMomentum;};
  
  recHitContainer hits(){ return _hits; };

  PTrajectoryStateOnDet trajectoryState(){return *PTraj;};
  TrajectorySeed TrajSeed(){return TrajectorySeed(trajectoryState(),hits(),direction());};
 private:

  bool construct( const TrackingRecHit* outerHit,
		  const TrackingRecHit* innerHit,
		  const GlobalPoint& vertexPos,
		  const GlobalError& vertexErr,
		  const edm::EventSetup& iSetup,
		  const edm::ParameterSet& p
		  );

  CurvilinearTrajectoryError initialError(
		   const GlobalPoint& vertexPos, const GlobalError& vertexErr, float sinTheta);

  TrajectoryStateTransform transformer;
  TransientTrackingRecHit::ConstRecHitPointer outrhit;
  TransientTrackingRecHit::ConstRecHitPointer intrhit;
  PropagationDirection _dir;
  boost::shared_ptr<PTrajectoryStateOnDet> PTraj;
  recHitContainer _hits;
  bool isValid_;

};

#endif 

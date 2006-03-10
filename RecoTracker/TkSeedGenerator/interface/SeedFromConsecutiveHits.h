#ifndef SeedFromConsecutiveHits_H
#define SeedFromConsecutiveHits_H

/** \class SeedFromConsecutiveHits
 * Seed constructed from the outer and inner RecHit (order important)
 * and the vertex constraints
 */
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
class DetLayer;
class SeedFromConsecutiveHits{
 public:
  
  // constructor in case the RecHits contain layer pointers.
  /*   SeedFromConsecutiveHits( const SiPixelRecHit& outerHit, */
  /* 			   const SiPixelRecHit& innerHit, */
  /* 			   const GlobalPoint& vertexPos, */
  /* 			   const GlobalError& vertexErr); */
  
  // constructor in case the RecHits do not contain layer pointers.
  SeedFromConsecutiveHits( const TrackingRecHit& outerHit,
			   const TrackingRecHit& innerHit,
			   const GlobalPoint& vertexPos,
			   const GlobalError& vertexErr,
			   const edm::EventSetup& iSetup);

  virtual  ~SeedFromConsecutiveHits(){};

  
  PropagationDirection direction(){
    //as in ORCA
    return alongMomentum;};
  

  edm::OwnVector<TrackingRecHit> hits(){
    return _hits;
  };

  PTrajectoryStateOnDet trajectoryState(){return *PTraj;};
  TrajectorySeed *TrajSeed(){return new TrajectorySeed(trajectoryState(),hits(),direction());};
 private:
  TrajectoryMeasurement theInnerMeas;
  TrajectoryMeasurement theOuterMeas;
  TransientTrackingRecHitBuilder TTRHBuilder;
  void construct( const TrackingRecHit& outerHit,
		  const TrackingRecHit& innerHit,
		  const GlobalPoint& vertexPos,
		  const GlobalError& vertexErr,
		  const edm::EventSetup& iSetup);

  CurvilinearTrajectoryError initialError( const TrackingRecHit& outerHit,
					   const TrackingRecHit& innerHit,
					   const GlobalPoint& vertexPos,
					   const GlobalError& vertexErr);

  TrajectoryStateTransform transformer;
  const TransientTrackingRecHit* outrhit;
  const TransientTrackingRecHit* intrhit;
  PropagationDirection _dir;
  PTrajectoryStateOnDet* PTraj;
  edm::OwnVector<TrackingRecHit> _hits;

};

#endif 

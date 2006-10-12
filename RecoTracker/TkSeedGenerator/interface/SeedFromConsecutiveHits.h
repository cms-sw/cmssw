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
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

class DetLayer;
class SeedFromConsecutiveHits{
 public:
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  // constructor in case the RecHits contain layer pointers.
  /*   SeedFromConsecutiveHits( const SiPixelRecHit& outerHit, */
  /* 			   const SiPixelRecHit& innerHit, */
  /* 			   const GlobalPoint& vertexPos, */
  /* 			   const GlobalError& vertexErr); */
  
  // constructor in case the RecHits do not contain layer pointers.
  SeedFromConsecutiveHits( const TrackingRecHit* outerHit,
			   const TrackingRecHit* innerHit,
			   const GlobalPoint& vertexPos,
			   const GlobalError& vertexErr,
			   const edm::EventSetup& iSetup,
			   const edm::ParameterSet& p
			   );

  virtual  ~SeedFromConsecutiveHits(){};

  
  PropagationDirection direction(){
    //as in ORCA
    return alongMomentum;};
  
  recHitContainer hits(){
    return _hits;
  };

 /*  edm::OwnVector<TrackingRecHit> hits(){ */
/*     return _hits; */
/*   }; */

  PTrajectoryStateOnDet trajectoryState(){return *PTraj;};
  TrajectorySeed TrajSeed(){return TrajectorySeed(trajectoryState(),hits(),direction());};
 private:
  //TrajectoryMeasurement theInnerMeas;
  //TrajectoryMeasurement theOuterMeas;

  void construct( const TrackingRecHit* outerHit,
		  const TrackingRecHit* innerHit,
		  const GlobalPoint& vertexPos,
		  const GlobalError& vertexErr,
		  const edm::EventSetup& iSetup,
		  const edm::ParameterSet& p
		  );

  CurvilinearTrajectoryError initialError( const TrackingRecHit* outerHit,
					   const TrackingRecHit* innerHit,
					   const GlobalPoint& vertexPos,
					   const GlobalError& vertexErr);

  TrajectoryStateTransform transformer;
  TransientTrackingRecHit::ConstRecHitPointer outrhit;
  TransientTrackingRecHit::ConstRecHitPointer intrhit;
  PropagationDirection _dir;
  boost::shared_ptr<PTrajectoryStateOnDet> PTraj;
  recHitContainer _hits;

};

#endif 

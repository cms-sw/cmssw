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

class DetLayer;
class SeedFromConsecutiveHits : 
public TrajectorySeed{
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
  //MP
  //  virtual const FreeTrajectoryState& freeTrajectoryState() const;
  //MP
  //  virtual PropagationDirection direction() const;
  // virtual LayerContainer layers() const;

/*   virtual bool share( const BasicTrajectorySeed&) const; */
/*  virtual BasicTrajectorySeed* clone() const; */


  //{return std::make_pair<outrhit,outrhit>;};
/*   std::vector<TrajectoryMeasurement> measurements() const;  */
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
 
  const TransientTrackingRecHit* outrhit;
  const TransientTrackingRecHit* intrhit;
};

#endif 

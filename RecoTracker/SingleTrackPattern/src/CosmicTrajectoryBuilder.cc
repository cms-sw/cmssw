//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia
#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

#include "DataFormats/Common/interface/OwnVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

CosmicTrajectoryBuilder::CosmicTrajectoryBuilder(const edm::ParameterSet& conf) : conf_(conf) { 

}


CosmicTrajectoryBuilder::~CosmicTrajectoryBuilder() {
}

void CosmicTrajectoryBuilder::init(const edm::EventSetup& es){

  es.get<IdealMagneticFieldRecord>().get(magfield);
  thePropagator= new AnalyticalPropagator(&(*magfield), alongMomentum);
}

void CosmicTrajectoryBuilder::run(const TrajectorySeedCollection &collseed,
				  const SiStripRecHit2DLocalPosCollection &collstereo,
				  const SiStripRecHit2DLocalPosCollection &collrphi ,
				  const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const edm::EventSetup& es,
				  TrackCandidateCollection &output,
				  const TrackerGeometry& tracker)
{

  TrajectorySeedCollection::const_iterator iseed;
  for(iseed=collseed.begin();iseed!=collseed.end();iseed++){
    Trajectory startingTraj = createStartingTrajectory(*iseed,tracker);
  }
 
};

Trajectory CosmicTrajectoryBuilder::createStartingTrajectory( const TrajectorySeed& seed,
					 const TrackerGeometry& tracker) const
{
 
  Trajectory result( seed, seed.direction());

  std::vector<TM> seedMeas = seedMeasurements(seed,tracker);
  if ( !seedMeas.empty()) {
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);
    }
  }
  return result;
}


std::vector<TrajectoryMeasurement> 
CosmicTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed,
					  const TrackerGeometry& tracker) const
{
 
  std::vector<TrajectoryMeasurement> result;
  TkTransientTrackingRecHitBuilder TTTRHBuilder(&tracker);
  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    TransientTrackingRecHit* recHit = TTTRHBuilder.build(&(*ihit));
    const GeomDet* hitGeomDet = tracker.idToDet( ihit->geographicalId());
    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));
    if (ihit == hitRange.second - 1) {
      PTrajectoryStateOnDet pState( seed.startingState());
      const GeomDet* gdet  = tracker.idToDet(DetId(pState.detId()));
      if (&gdet->surface() != &hitGeomDet->surface()) {
	edm::LogError ("Propagation")<< "CosmicTrajectoryBuilder error: the seed state is not on the surface of the detector of the last seed hit";
	return std::vector<TrajectoryMeasurement>(); // FIXME: should throw exception
      }
      TSOS updatedState= tsTransform.transientState( pState, &(gdet->surface()), 
						     &(*magfield));
  
      result.push_back(TM( invalidState, updatedState, recHit));
    } 
    else {
      result.push_back(TM( invalidState, recHit));
    }
    
  }
 
  return result;
}

#ifndef RoadSearchPairLess_h
#define RoadSearchPairLess_h

//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMakerAlgorithm
// 
// Description:     Converts cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/11/08 14:25:50 $
// $Revision: 1.4 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

class RoadSearchPairLess
{
 public:
  
  RoadSearchPairLess(){ };

  bool operator()(const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM1 ,
		  const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM2 ) const
  {
    return
      InsideOutCompare(HitTM1,HitTM2);
  }  
 

 private:

   bool InsideOutCompare( const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM1 ,
			  const std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> HitTM2 )  const;

};

#endif

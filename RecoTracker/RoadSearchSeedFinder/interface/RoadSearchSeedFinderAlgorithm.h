#ifndef RoadSearchSeedFinderAlgorithm_h
#define RoadSearchSeedFinderAlgorithm_h

//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinderAlgorithm
// 
// Description:     Loops over Roads, checks for every
//                  RoadSeed if hits are in the inner and
//                  outer SeedRing, applies cuts for all 
//                  combinations of inner and outer SeedHits,
//                  stores valid combination in TrackingSeed
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/28 22:54:06 $
// $Revision: 1.3 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"

class RoadSearchSeedFinderAlgorithm 
{
 public:
  
  RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchSeedFinderAlgorithm();

  /// Runs the algorithm
  void run(const edm::Handle<SiStripRecHit2DMatchedLocalPosCollection> &handle,
           const edm::Handle<SiStripRecHit2DLocalPosCollection> &handle2,
	   const edm::EventSetup& es,
	   TrajectorySeedCollection &output);

  CurvilinearTrajectoryError initialError( const TrackingRecHit* outerHit,
					   const TrackingRecHit* innerHit,
					   const GlobalPoint& vertexPos,
					   const GlobalError& vertexErr);
 private:
  edm::ParameterSet conf_;

};

#endif

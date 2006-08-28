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
// $Author: burkett $
// $Date: 2006/08/25 16:20:31 $
// $Revision: 1.7 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

class RoadSearchSeedFinderAlgorithm 
{
 public:
  
  RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchSeedFinderAlgorithm();

  // Runs the algorithm
  void run(const SiStripRecHit2DCollection* rphiRecHits,
	   const SiStripRecHit2DCollection* stereoRecHits,
	   const SiStripMatchedRecHit2DCollection* matchedRecHits,
	   const SiPixelRecHitCollection* pixelRecHits,
	   const edm::EventSetup& es,
	   TrajectorySeedCollection &output);
  
  CurvilinearTrajectoryError initialError( const TrackingRecHit* outerHit,
					   const TrackingRecHit* innerHit,
					   const GlobalPoint& vertexPos,
					   const GlobalError& vertexErr);

  TrajectorySeed makeSeedFromPair(const TrackingRecHit* innerHit,
				  const GlobalPoint* innerPos,
				  const TrackingRecHit* outerHit,
				  const GlobalPoint* outerPos,
				  const edm::EventSetup& es);

  void makeSeedsFromInnerHit(TrajectorySeedCollection* outcoll,
			     const TrackingRecHit* innerHit,
			     const edm::OwnVector<TrackingRecHit>* outerHits,
			     const TrackerGeometry *tracker,const edm::EventSetup& es);


 private:
  edm::ParameterSet conf_;

};

#endif

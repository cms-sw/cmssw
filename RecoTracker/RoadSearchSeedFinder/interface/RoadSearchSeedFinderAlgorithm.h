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
// $Date: 2007/03/01 08:16:17 $
// $Revision: 1.14 $
//

#include <string>
#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchCircleSeed.h"

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
  
  CurvilinearTrajectoryError initialError( const GlobalPoint& vertexPos,
					   const GlobalError& vertexErr);

  bool convertCircleToTrajectorySeed(TrajectorySeedCollection &output,
				     RoadSearchCircleSeed circleSeed,
				     const edm::EventSetup& es);

  bool mergeCircleSeeds(std::vector<RoadSearchCircleSeed> &circleSeeds);
  
  bool calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
				    GlobalPoint ring1GlobalPoint,
				    TrackingRecHit *ring1RecHit,
				    std::vector<TrackingRecHit*> ring2RecHits,
				    std::vector<TrackingRecHit*> ring3RecHits);

  bool calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
				    GlobalPoint ring1GlobalPoint,
				    TrackingRecHit *ring1RecHit,
				    std::vector<TrackingRecHit*> ring2RecHits);

  bool calculateCircleSeedsFromRingsOneInnerTwoOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
						     const Ring* ring1,
						     const Ring* ring2,
						     const Ring* ring3);

  bool calculateCircleSeedsFromRingsTwoInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
						     const Ring* ring1,
						     const Ring* ring2,
						     const Ring* ring3);
  bool calculateCircleSeedsFromRingsOneInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
						     const Ring* ring1,
						     const Ring* ring2);

  bool ringsOnSameLayer(const Ring *ring1, 
			const Ring* ring2);
  bool detIdsOnSameLayer(DetId id1, 
			 DetId id2);

 private:

  double       minPt_;
  double       maxImpactParameter_;
  double       phiRangeDetIdLookup_;
  double       mergeSeedsCenterCut_;
  double       mergeSeedsRadiusCut_;
  unsigned int mergeSeedsDifferentHitsCut_;

  DetHitAccess innerSeedHitVector_;
  DetHitAccess outerSeedHitVector_;

  DetHitAccess::accessMode innerSeedHitAccessMode_;
  bool                     innerSeedHitAccessUseRPhi_;
  bool                     innerSeedHitAccessUseStereo_;

  DetHitAccess::accessMode outerSeedHitAccessMode_;
  bool                     outerSeedHitAccessUseRPhi_;
  bool                     outerSeedHitAccessUseStereo_;

  const TrackerGeometry *tracker_;
  const Roads           *roads_;
  const MagneticField   *magnet_;

  std::ostringstream output_;

  double beamSpotZMagneticField_;
  double minRadius_;

  unsigned int compareLast_;
  double maxCenterDistance_;
  double maxRadiusDifference_;
  double maxCurvatureDifference_;
  unsigned int numMergedCircles_;

  std::vector<unsigned int> usedSeedRingCombinations_;

  std::string mode_;

  std::string roadsLabel_;

};

#endif

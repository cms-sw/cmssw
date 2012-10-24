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
// $Author: vlimant $
// $Date: 2009/01/26 10:15:10 $
// $Revision: 1.22 $
//

#include <string>
#include <sstream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

#include "RoadSearchCircleSeed.h"

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
	   RoadSearchSeedCollection &output);
  
  bool mergeCircleSeeds(std::vector<RoadSearchCircleSeed> &circleSeeds);
  
  bool calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
				    const Roads::RoadSeed *seed,
				    const Roads::RoadSet *set,
				    GlobalPoint ring1GlobalPoint,
				    TrackingRecHit *ring1RecHit,
				    std::vector<TrackingRecHit*> ring2RecHits,
				    std::vector<TrackingRecHit*> ring3RecHits);

  bool calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
				    const Roads::RoadSeed *seed,
				    const Roads::RoadSet *set,
				    GlobalPoint ring1GlobalPoint,
				    TrackingRecHit *ring1RecHit,
				    std::vector<TrackingRecHit*> ring2RecHits);

  bool calculateCircleSeedsFromRingsOneInnerTwoOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
						     const Roads::RoadSeed *seed,
						     const Roads::RoadSet *set,
						     const Ring* ring1,
						     const Ring* ring2,
						     const Ring* ring3);

  bool calculateCircleSeedsFromRingsTwoInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
						     const Roads::RoadSeed *seed,
						     const Roads::RoadSet *set,
						     const Ring* ring1,
						     const Ring* ring2,
						     const Ring* ring3);
  bool calculateCircleSeedsFromRingsOneInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
						     const Roads::RoadSeed *seed,
						     const Roads::RoadSet *set,
						     const Ring* ring1,
						     const Ring* ring2);

  bool ringsOnSameLayer(const Ring *ring1, 
			const Ring* ring2);
  bool detIdsOnSameLayer(DetId id1, 
			 DetId id2);
  
  unsigned int ClusterCounter(const edmNew::DetSetVector<SiStripCluster>* clusters);
  
 private:

  double       minPt_;
  double       maxBarrelImpactParameter_;
  double       maxEndcapImpactParameter_;
  double       phiRangeDetIdLookup_;

  double       mergeSeedsCenterCut_;
  double       mergeSeedsRadiusCut_;
  double       mergeSeedsCenterCut_A_;
  double       mergeSeedsRadiusCut_A_;
  double       mergeSeedsCenterCut_B_;
  double       mergeSeedsRadiusCut_B_;
  double       mergeSeedsCenterCut_C_;
  double       mergeSeedsRadiusCut_C_;
  unsigned int mergeSeedsDifferentHitsCut_;

  int          maxNumberOfSeeds_;

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

  std::vector<unsigned int> usedSeedRingCombinations_;

  std::string mode_;

  std::string roadsLabel_;

  //***top-bottom
  bool allPositiveOnly;
  bool allNegativeOnly;
  //***

};

#endif

//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchSeedFinderAlgorithm
// 
// Description:     Loops over Roads, checks for every
//                  RoadSeed if hits are in the inner and
//                  outer SeedRing, applies cuts for all 
//                  combinations of inner and outer SeedHits,
//                  stores valid combination in TrajectorySeed
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/01 08:16:18 $
// $Revision: 1.24 $
//

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinderAlgorithm.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

const double speedOfLight = 2.99792458e8;
const double unitCorrection = speedOfLight * 1e-2 * 1e-9;

RoadSearchSeedFinderAlgorithm::RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf) { 


  minPt_                      = conf.getParameter<double>("MinimalReconstructedTransverseMomentum");
  maxImpactParameter_         = conf.getParameter<double>("MaximalImpactParameter");
  phiRangeDetIdLookup_        = conf.getParameter<double>("PhiRangeForDetIdLookupInRings");
  compareLast_                = conf.getParameter<unsigned int>("MergeSeedsCompareLast");
  mergeSeedsCenterCut_        = conf.getParameter<double>("MergeSeedsCenterCut");
  mergeSeedsRadiusCut_        = conf.getParameter<double>("MergeSeedsRadiusCut");
  mergeSeedsDifferentHitsCut_ = conf.getParameter<unsigned int>("MergeSeedsDifferentHitsCut");
  mode_                       = conf.getParameter<std::string>("Mode");

  // safety check for mode
  if ( mode_ != "STANDARD" && mode_ != "COSMICS" ) {
    mode_ = "STANDARD";
  }

  std::string tmp             = conf.getParameter<std::string>("InnerSeedRecHitAccessMode");
  if ( tmp == "STANDARD" ) {
    innerSeedHitAccessMode_ = DetHitAccess::standard;
  } else if ( tmp == "RPHI" ) {
    innerSeedHitAccessMode_ = DetHitAccess::rphi;
  } else {
    innerSeedHitAccessMode_ = DetHitAccess::standard;
  }
  innerSeedHitAccessUseRPhi_  = conf.getParameter<bool>("InnerSeedRecHitAccessUseRPhi");
  innerSeedHitAccessUseStereo_  = conf.getParameter<bool>("InnerSeedRecHitAccessUseStereo");

  tmp                         = conf.getParameter<std::string>("OuterSeedRecHitAccessMode");
  if ( tmp == "STANDARD" ) {
    outerSeedHitAccessMode_ = DetHitAccess::standard;
  } else if ( tmp == "RPHI" ) {
    outerSeedHitAccessMode_ = DetHitAccess::rphi;
  } else {
    outerSeedHitAccessMode_ = DetHitAccess::standard;
  }
  outerSeedHitAccessUseRPhi_  = conf.getParameter<bool>("OuterSeedRecHitAccessUseRPhi");
  outerSeedHitAccessUseStereo_  = conf.getParameter<bool>("OuterSeedRecHitAccessUseStereo");

  // configure DetHitAccess
  innerSeedHitVector_.setMode(innerSeedHitAccessMode_);
  innerSeedHitVector_.use_rphiRecHits(innerSeedHitAccessUseRPhi_);
  innerSeedHitVector_.use_stereoRecHits(innerSeedHitAccessUseStereo_);
  outerSeedHitVector_.setMode(outerSeedHitAccessMode_);
  outerSeedHitVector_.use_rphiRecHits(outerSeedHitAccessUseRPhi_);
  outerSeedHitVector_.use_stereoRecHits(outerSeedHitAccessUseStereo_);

  roadsLabel_ = conf.getParameter<std::string>("RoadsLabel");

}

RoadSearchSeedFinderAlgorithm::~RoadSearchSeedFinderAlgorithm() {
}


void RoadSearchSeedFinderAlgorithm::run(const SiStripRecHit2DCollection* rphiRecHits,
					const SiStripRecHit2DCollection* stereoRecHits,
					const SiStripMatchedRecHit2DCollection* matchedRecHits,
					const SiPixelRecHitCollection* pixelRecHits,
					const edm::EventSetup& es,
					TrajectorySeedCollection &output)
{

  // initialize general hit access for road search
  innerSeedHitVector_.setCollections(rphiRecHits,stereoRecHits,matchedRecHits,pixelRecHits);
  outerSeedHitVector_.setCollections(rphiRecHits,stereoRecHits,matchedRecHits,pixelRecHits);

  // get roads
  edm::ESHandle<Roads> roads;
  es.get<RoadMapRecord>().get(roadsLabel_,roads);
  roads_ = roads.product();

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  tracker_ = tracker.product();

  // get magnetic field
  edm::ESHandle<MagneticField> magnet;
  es.get<IdealMagneticFieldRecord>().get(magnet);
  magnet_ = magnet.product();

  // get magnetic field for 0,0,0 , approximation for minRadius calculation
  beamSpotZMagneticField_ = magnet_->inTesla(GlobalPoint(0,0,0)).z();
  // calculate minimal radius at globalPoint in cm, take the z component of the magnetic field at GlobalPoint 2
  if ( beamSpotZMagneticField_ == 0 ) {
    minRadius_ = 999999999999.;
  } else {
    minRadius_ = minPt_ / 0.3 / beamSpotZMagneticField_ * 100;
  }

  // temporary storing collection of circle seeds
  std::vector<RoadSearchCircleSeed> circleSeeds;

  // loop over seed Ring pairs
  for ( Roads::const_iterator road = roads_->begin(); road != roads_->end(); ++road ) {

    Roads::RoadSeed seed = (*road).first;

    maxCenterDistance_   = 0.;
    maxRadiusDifference_ = 0.;
    maxCurvatureDifference_ = 0.;
    numMergedCircles_    = 0;

    
    if ( mode_ == "COSMICS" ) {
      // loop over seed ring pairs
      // draw straight line
      for ( std::vector<const Ring*>::const_iterator innerSeedRing = seed.first.begin();
	    innerSeedRing != seed.first.end();
	    ++innerSeedRing) {
	for ( std::vector<const Ring*>::const_iterator outerSeedRing = seed.second.begin();
	      outerSeedRing != seed.second.end();
	      ++outerSeedRing) {
	  calculateCircleSeedsFromRingsOneInnerOneOuter(circleSeeds,
							*innerSeedRing,
							*outerSeedRing);
	  
	}
      }
    } else if ( mode_ == "STANDARD" ) {

      // take combinations of one inner and two outer or two inner and one outer seed ring
      for ( std::vector<const Ring*>::const_iterator innerSeedRing1 = seed.first.begin();
	    innerSeedRing1 != seed.first.end();
	    ++innerSeedRing1) {
	// two inner, one outer
	for ( std::vector<const Ring*>::const_iterator innerSeedRing2 = innerSeedRing1+1;
	      innerSeedRing2 != seed.first.end();
	      ++innerSeedRing2) {
	  if ( !ringsOnSameLayer(*innerSeedRing1,*innerSeedRing2) ) {
	    for ( std::vector<const Ring*>::const_iterator outerSeedRing = seed.second.begin();
		  outerSeedRing != seed.second.end();
		  ++outerSeedRing) {
	      // calculate seed ring combination identifier
	      unsigned int identifier = (*innerSeedRing1)->getindex() * 1000000 +
		(*innerSeedRing2)->getindex() * 1000 +
		(*outerSeedRing)->getindex();
	      bool check = true;
	      for ( std::vector<unsigned int>::iterator alreadyChecked = usedSeedRingCombinations_.begin();
		    alreadyChecked != usedSeedRingCombinations_.end();
		    ++alreadyChecked ) {
		if ( identifier == *alreadyChecked ) {
		  check = false;
		  break;
		}
	      }

	      if ( check ) {
		usedSeedRingCombinations_.push_back(identifier);
		calculateCircleSeedsFromRingsTwoInnerOneOuter(circleSeeds,
							      *innerSeedRing1,
							      *innerSeedRing2,
							      *outerSeedRing);
	      }
	    }	  
	  }
	}
	// one inner, two outer
	for ( std::vector<const Ring*>::const_iterator outerSeedRing1 = seed.second.begin();
	      outerSeedRing1 != seed.second.end();
	      ++outerSeedRing1) {
	  for ( std::vector<const Ring*>::const_iterator outerSeedRing2 = outerSeedRing1+1;
		outerSeedRing2 != seed.second.end();
		++outerSeedRing2) {
	    if ( !ringsOnSameLayer(*outerSeedRing1,*outerSeedRing2) ) {
	      // calculate seed ring combination identifier
	      unsigned int identifier = (*innerSeedRing1)->getindex() * 1000000 +
		(*outerSeedRing1)->getindex() * 1000 +
		(*outerSeedRing2)->getindex();
	      bool check = true;
	      for ( std::vector<unsigned int>::iterator alreadyChecked = usedSeedRingCombinations_.begin();
		    alreadyChecked != usedSeedRingCombinations_.end();
		    ++alreadyChecked ) {
		if ( identifier == *alreadyChecked ) {
		  check = false;
		  break;
		}
	      }

	      if ( check ) {
		usedSeedRingCombinations_.push_back(identifier);
		calculateCircleSeedsFromRingsOneInnerTwoOuter(circleSeeds,
							      *innerSeedRing1,
							      *outerSeedRing1,
							      *outerSeedRing2);
	      }
	    }
	  }
	}
      }
    }

    for ( std::vector<RoadSearchCircleSeed>::iterator circle = circleSeeds.begin();
	  circle != circleSeeds.end();
	  ++circle ) {
//       output_ << circle->print();
      convertCircleToTrajectorySeed(output,*circle,es);
    }

    circleSeeds.clear();
    usedSeedRingCombinations_.clear();

//     edm::LogInfo("RoadSearch") << output_.str() << std::endl; 
//     output_.str("");
  }

  edm::LogInfo("RoadSearch") << "Found " << output.size() << " seeds."; 

}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromRingsTwoInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
								       const Ring* ring1,
								       const Ring* ring2,
								       const Ring* ring3) {
  //
  // calculate RoadSearchCircleSeed
  //
  // apply circle seed cuts
  //

  // return value
  bool result = true;

  // loop over detid's in first rings
  for ( Ring::const_iterator ring1DetIdIterator = ring1->begin(); 
	ring1DetIdIterator != ring1->end(); 
	++ring1DetIdIterator ) {

    DetId ring1DetId = ring1DetIdIterator->second;
    std::vector<TrackingRecHit*> ring1RecHits = innerSeedHitVector_.getHitVector(&ring1DetId);

    // loop over inner rechits
    for (std::vector<TrackingRecHit*>::const_iterator ring1RecHit = ring1RecHits.begin();
	 ring1RecHit != ring1RecHits.end(); 
	 ++ring1RecHit) {
	    
      GlobalPoint ring1GlobalPoint = tracker_->idToDet((*ring1RecHit)->geographicalId())->surface().toGlobal((*ring1RecHit)->localPosition());

      // calculate phi range around inner hit
      double innerphi = ring1GlobalPoint.phi();
      double upperPhiRangeBorder = innerphi + phiRangeDetIdLookup_;
      double lowerPhiRangeBorder = innerphi - phiRangeDetIdLookup_;
      if (upperPhiRangeBorder>Geom::pi()) upperPhiRangeBorder -= Geom::twoPi();
      if (lowerPhiRangeBorder<(-Geom::pi())) lowerPhiRangeBorder += Geom::twoPi();

      // retrieve vectors of TrackingRecHits in ring2 and ring3 in phi range
      std::vector<TrackingRecHit*> ring2RecHits;
      std::vector<TrackingRecHit*> ring3RecHits;

      if (lowerPhiRangeBorder <= upperPhiRangeBorder ) {
	for ( Ring::const_iterator outerRingDetId = ring2->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring2->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring3->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring3->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring3RecHits.insert(ring3RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
      } else {
	for ( Ring::const_iterator outerRingDetId = ring2->begin(); 
	      outerRingDetId != ring2->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring3->begin(); 
	      outerRingDetId != ring3->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring3RecHits.insert(ring3RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring2->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring2->end();
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring3->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring3->end();
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring3RecHits.insert(ring3RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
      }

      if ( ring2RecHits.size() > 0 &&
	   ring3RecHits.size() > 0 ) {
	calculateCircleSeedsFromHits(circleSeeds, ring1GlobalPoint, *ring1RecHit, ring2RecHits, ring3RecHits);
      }

    }
  }

  return result;

}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromRingsOneInnerTwoOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
								       const Ring* ring1,
								       const Ring* ring2,
								       const Ring* ring3) {
  //
  // calculate RoadSearchCircleSeed
  //
  // apply circle seed cuts
  //

  // return value
  bool result = true;

  // loop over detid's in first rings
  for ( Ring::const_iterator ring1DetIdIterator = ring1->begin(); 
	ring1DetIdIterator != ring1->end(); 
	++ring1DetIdIterator ) {

    DetId ring1DetId = ring1DetIdIterator->second;
    std::vector<TrackingRecHit*> ring1RecHits = innerSeedHitVector_.getHitVector(&ring1DetId);

    // loop over inner rechits
    for (std::vector<TrackingRecHit*>::const_iterator ring1RecHit = ring1RecHits.begin();
	 ring1RecHit != ring1RecHits.end(); 
	 ++ring1RecHit) {
	    
      GlobalPoint ring1GlobalPoint = tracker_->idToDet((*ring1RecHit)->geographicalId())->surface().toGlobal((*ring1RecHit)->localPosition());

      // calculate phi range around inner hit
      double innerphi = ring1GlobalPoint.phi();
      double upperPhiRangeBorder = innerphi + phiRangeDetIdLookup_;
      double lowerPhiRangeBorder = innerphi - phiRangeDetIdLookup_;
      if (upperPhiRangeBorder>Geom::pi()) upperPhiRangeBorder -= Geom::twoPi();
      if (lowerPhiRangeBorder<(-Geom::pi())) lowerPhiRangeBorder += Geom::twoPi();

      // retrieve vectors of TrackingRecHits in ring2 and ring3 in phi range
      std::vector<TrackingRecHit*> ring2RecHits;
      std::vector<TrackingRecHit*> ring3RecHits;

      if (lowerPhiRangeBorder <= upperPhiRangeBorder ) {
	for ( Ring::const_iterator outerRingDetId = ring2->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring2->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring3->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring3->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring3RecHits.insert(ring3RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
      } else {
	for ( Ring::const_iterator outerRingDetId = ring2->begin(); 
	      outerRingDetId != ring2->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring3->begin(); 
	      outerRingDetId != ring3->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring3RecHits.insert(ring3RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring2->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring2->end();
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring3->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring3->end();
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring3RecHits.insert(ring3RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
      }

      if ( ring2RecHits.size() > 0 &&
	   ring3RecHits.size() > 0 ) {
	calculateCircleSeedsFromHits(circleSeeds, ring1GlobalPoint, *ring1RecHit, ring2RecHits, ring3RecHits);
      }

    }
  }

  return result;

}
								       
bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromRingsOneInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
										  const Ring* ring1,
										  const Ring* ring2) {
  //
  // calculate RoadSearchCircleSeed
  //
  // apply circle seed cuts
  //

  // return value
  bool result = true;

  // loop over detid's in first rings
  for ( Ring::const_iterator ring1DetIdIterator = ring1->begin(); 
	ring1DetIdIterator != ring1->end(); 
	++ring1DetIdIterator ) {

    DetId ring1DetId = ring1DetIdIterator->second;
    std::vector<TrackingRecHit*> ring1RecHits = innerSeedHitVector_.getHitVector(&ring1DetId);

    // loop over inner rechits
    for (std::vector<TrackingRecHit*>::const_iterator ring1RecHit = ring1RecHits.begin();
	 ring1RecHit != ring1RecHits.end(); 
	 ++ring1RecHit) {
	    
      GlobalPoint ring1GlobalPoint = tracker_->idToDet((*ring1RecHit)->geographicalId())->surface().toGlobal((*ring1RecHit)->localPosition());

      // calculate phi range around inner hit
      double innerphi = ring1GlobalPoint.phi();
      double upperPhiRangeBorder = innerphi + phiRangeDetIdLookup_;
      double lowerPhiRangeBorder = innerphi - phiRangeDetIdLookup_;
      if (upperPhiRangeBorder>Geom::pi()) upperPhiRangeBorder -= Geom::twoPi();
      if (lowerPhiRangeBorder<(-Geom::pi())) lowerPhiRangeBorder += Geom::twoPi();

      // retrieve vectors of TrackingRecHits in ring2 in phi range
      std::vector<TrackingRecHit*> ring2RecHits;

      if (lowerPhiRangeBorder <= upperPhiRangeBorder ) {
	for ( Ring::const_iterator outerRingDetId = ring2->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring2->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
      } else {
	for ( Ring::const_iterator outerRingDetId = ring2->begin(); 
	      outerRingDetId != ring2->upper_bound(upperPhiRangeBorder);
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
	for ( Ring::const_iterator outerRingDetId = ring2->lower_bound(lowerPhiRangeBorder); 
	      outerRingDetId != ring2->end();
	      ++outerRingDetId) {
	  std::vector<TrackingRecHit*> rings = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	  ring2RecHits.insert(ring2RecHits.end(),
			      rings.begin(),
			      rings.end());
	}
      }

      if ( ring2RecHits.size() > 0 ) {
// 	output_ << "Combination with inner and outer hits, ring 1: " << ring1->getindex() << " ring 2: " << ring2->getindex() << "\n";
	calculateCircleSeedsFromHits(circleSeeds, ring1GlobalPoint, *ring1RecHit, ring2RecHits);
      }

    }
  }

  return result;

}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
								 GlobalPoint ring1GlobalPoint,
								 TrackingRecHit *ring1RecHit,
								 std::vector<TrackingRecHit*> ring2RecHits,
								 std::vector<TrackingRecHit*> ring3RecHits) {
  //
  // calculate RoadSearchCircleSeed
  //
  // apply circle seed cuts
  //

  // return value
  bool result = true;

  for ( std::vector<TrackingRecHit*>::iterator ring2RecHit = ring2RecHits.begin();
	ring2RecHit != ring2RecHits.end();
	++ring2RecHit) {
    GlobalPoint ring2GlobalPoint = tracker_->idToDet((*ring2RecHit)->geographicalId())->surface().toGlobal((*ring2RecHit)->localPosition());
    for ( std::vector<TrackingRecHit*>::iterator ring3RecHit = ring3RecHits.begin();
	  ring3RecHit != ring3RecHits.end();
	  ++ring3RecHit) {
      GlobalPoint ring3GlobalPoint = tracker_->idToDet((*ring3RecHit)->geographicalId())->surface().toGlobal((*ring3RecHit)->localPosition());

      RoadSearchCircleSeed circle(ring1RecHit,
				  *ring2RecHit,
				  *ring3RecHit,
				  ring1GlobalPoint,
				  ring2GlobalPoint,
				  ring3GlobalPoint);

//       output_ << circle.print();

      bool addCircle = false;
      if ( circle.Type() == RoadSearchCircleSeed::straightLine ) {
	addCircle = true;
      } else {
	if ( (circle.Radius() > minRadius_) &&
	     (circle.ImpactParameter() < maxImpactParameter_) ) {
	  addCircle = true;
	}

	// do merging if compareLast > 0
	if ( compareLast_ > 0 ) {

	  std::vector<RoadSearchCircleSeed>::iterator begin;
	  if ( (compareLast_ == 9999999) ||
	       (compareLast_ > circleSeeds.size()) ) {
	    begin = circleSeeds.begin();
	  } else {
	    begin = circleSeeds.end()-(compareLast_+1);
	  }
	  std::vector<RoadSearchCircleSeed>::iterator end = circleSeeds.end();


	  for (std::vector<RoadSearchCircleSeed>::iterator alreadyContainedCircle = begin;
	       alreadyContainedCircle != end;
	       ++alreadyContainedCircle ) {
	    // cut on percentage of distance between centers vs. average of centers
	    double averageCenter = std::sqrt(((alreadyContainedCircle->Center().x()+circle.Center().x())/2) *
					     ((alreadyContainedCircle->Center().x()+circle.Center().x())/2) +
					     ((alreadyContainedCircle->Center().y()+circle.Center().y())/2) *
					     ((alreadyContainedCircle->Center().y()+circle.Center().y())/2) );
	    double differenceCenter = std::sqrt((alreadyContainedCircle->Center().x()-circle.Center().x()) *
						(alreadyContainedCircle->Center().x()-circle.Center().x()) +
						(alreadyContainedCircle->Center().y()-circle.Center().y()) *
						(alreadyContainedCircle->Center().y()-circle.Center().y()));
	    double percentageCenter = differenceCenter / averageCenter;
	    // cut on percentage of difference of radii vs, average of radii
	    double averageRadius = (alreadyContainedCircle->Radius() + circle.Radius() ) /2;
	    double differenceRadius = std::abs(alreadyContainedCircle->Radius() - circle.Radius());
	    double percentageRadius = differenceRadius / averageRadius;
	    if ( (percentageCenter < mergeSeedsCenterCut_) &&
		 (percentageRadius < mergeSeedsRadiusCut_) ) {
	      
	      addCircle = false;
	      break;
	    }
	  }
	}
      }


      if ( addCircle ) {
	circleSeeds.push_back(circle);
      }
    }
  }

  return result;
}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
								 GlobalPoint ring1GlobalPoint,
								 TrackingRecHit *ring1RecHit,
								 std::vector<TrackingRecHit*> ring2RecHits) {
  //
  // calculate RoadSearchCircleSeed from two hits, calculate straight line
  //
  //

  // return value
  bool result = true;

  for ( std::vector<TrackingRecHit*>::iterator ring2RecHit = ring2RecHits.begin();
	ring2RecHit != ring2RecHits.end();
	++ring2RecHit) {
    GlobalPoint ring2GlobalPoint = tracker_->idToDet((*ring2RecHit)->geographicalId())->surface().toGlobal((*ring2RecHit)->localPosition());

    RoadSearchCircleSeed circle(ring1RecHit,
				*ring2RecHit,
				ring1GlobalPoint,
				ring2GlobalPoint);
    
//     output_ << circle.print();


    circleSeeds.push_back(circle);
  }

  return result;
}

bool RoadSearchSeedFinderAlgorithm::convertCircleToTrajectorySeed(TrajectorySeedCollection &output,
									    RoadSearchCircleSeed circleSeed,
									    const edm::EventSetup& es) {
  //
  // convert circle seed to trajectory seed
  // FIXME: deactivated for now
  //

  // return value
  bool result = true;

//   // use correct tk seed generator from consecutive hits
//   GlobalTrackingRegion region;
//   GlobalError vtxerr( std::sqrt(region.originRBound()),
// 		      0, std::sqrt(region.originRBound()),
// 		      0, 0, std::sqrt(region.originZBound()));

//   // make PTrajectoryOnState from three hits
//   std::vector<GlobalPoint> points = circleSeed.Points();
//   if ( points.size() >= 3 ) {
//     std::vector<TrackingRecHit*> recHits = circleSeed.Hits();

//     FastHelix helix(points[2],
// 		    points[1],
// 		    points[0],
// 		    es);
      
//     FreeTrajectoryState fts( helix.stateAtVertex().parameters(),
// 			     initialError(region.origin(), vtxerr));
    
//     AnalyticalPropagator  thePropagator(magnet_, alongMomentum);
    
//     const TrajectoryStateOnSurface innerState = thePropagator.propagate(fts,
// 									tracker_->idToDet(recHits[0]->geographicalId())->surface());
      
//     if (innerState.isValid()){
// 	//
// 	// create the OwnVector of TrackingRecHits
// 	edm::OwnVector<TrackingRecHit> rh;
	
// 	for (std::vector<TrackingRecHit*>::iterator hit = recHits.begin();
// 	     hit != recHits.end();
// 	     ++hit ) {
// 	  rh.push_back((*hit)->clone());
// 	}
// 	TrajectoryStateTransform transformer;
	
// 	PTrajectoryStateOnDet * PTraj=  transformer.persistentState(innerState, recHits[0]->geographicalId().rawId());
// 	TrajectorySeed ts(*PTraj,rh,alongMomentum);
// 	delete PTraj;  

// 	output.push_back(ts);
//     }
//   }
  

  std::vector<TrackingRecHit*> recHits = circleSeed.Hits();

  // create the OwnVector of TrackingRecHits
  edm::OwnVector<TrackingRecHit> rh;
  for (std::vector<TrackingRecHit*>::iterator hit = recHits.begin();
       hit != recHits.end();
       ++hit ) {
    rh.push_back((*hit)->clone());
  }
  PTrajectoryStateOnDet PTraj;
  TrajectorySeed ts(PTraj,rh,alongMomentum);
  output.push_back(ts);
  
  return result;

}

CurvilinearTrajectoryError RoadSearchSeedFinderAlgorithm::
initialError( const GlobalPoint& vertexPos,
	      const GlobalError& vertexErr)
{
  AlgebraicSymMatrix C(5,1);

  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr;

  return CurvilinearTrajectoryError(C);
}

bool RoadSearchSeedFinderAlgorithm::ringsOnSameLayer(const Ring *ring1, const Ring* ring2) {
  //
  // check whether two input rings are on the same layer
  //

  // return value
  bool result = false;

  // get first DetId of ring
  const DetId ring1DetId = ring1->getFirst();
  const DetId ring2DetId = ring2->getFirst();

  result = detIdsOnSameLayer(ring1DetId,ring2DetId);
  
  return result;
}

bool RoadSearchSeedFinderAlgorithm::detIdsOnSameLayer(DetId id1, DetId id2) {
  //
  // check whether two detids are on the same layer
  //

  // return value
  bool result = false;

  // check if both rings belong to same subdetector
  if ( (unsigned int)id1.subdetId() == StripSubdetector::TIB && 
       (unsigned int)id2.subdetId() == StripSubdetector::TIB ) {
    // make TIBDetId instance
    TIBDetId id1TIB(id1.rawId());
    TIBDetId id2TIB(id2.rawId());
    // check whether both rings are on the same TIB layer
    if ( id1TIB.layer() == id2TIB.layer() ) {
      result = true;
    }
  } else if ( (unsigned int)id1.subdetId() == StripSubdetector::TOB &&
	      (unsigned int)id2.subdetId() == StripSubdetector::TOB ) {
    // make TOBDetId instance
    TOBDetId id1TOB(id1.rawId());
    TOBDetId id2TOB(id2.rawId());
    // check whether both rings are on the same TOB layer
    if ( id1TOB.layer() == id2TOB.layer() ) {
      result = true;
    }
  } else if ( (unsigned int)id1.subdetId() == StripSubdetector::TID && 
	      (unsigned int)id2.subdetId() == StripSubdetector::TID) {
    // make TIDDetId instance
    TIDDetId id1TID(id1.rawId());
    TIDDetId id2TID(id2.rawId());
    // check whether both rings are on the same TID wheel
    if ( id1TID.wheel() == id2TID.wheel() ) {
      result = true;
    }
  } else if ( (unsigned int)id1.subdetId() == StripSubdetector::TEC &&
	      (unsigned int)id2.subdetId() == StripSubdetector::TEC ) {
    // make TECDetId instance
    TECDetId id1TEC(id1.rawId());
    TECDetId id2TEC(id2.rawId());
    // check whether both rings are on the same TEC wheel
    if ( id1TEC.wheel() == id2TEC.wheel() ) {
      result = true;
    }
  } else if ( (unsigned int)id1.subdetId() == PixelSubdetector::PixelBarrel && 
	      (unsigned int)id2.subdetId() == PixelSubdetector::PixelBarrel) {
    // make PXBDetId instance
    PXBDetId id1PXB(id1.rawId());
    PXBDetId id2PXB(id2.rawId());
    // check whether both rings are on the same PXB layer
    if ( id1PXB.layer() == id2PXB.layer() ) {
      result = true;
    }
  } else if ( (unsigned int)id1.subdetId() == PixelSubdetector::PixelEndcap &&
	      (unsigned int)id2.subdetId() == PixelSubdetector::PixelEndcap) {
    // make PXFDetId instance
    PXFDetId id1PXF(id1.rawId());
    PXFDetId id2PXF(id2.rawId());
    // check whether both rings are on the same PXF disk
    if ( id1PXF.disk() == id2PXF.disk() ) {
      result = true;
    }
  }
  
  return result;
}

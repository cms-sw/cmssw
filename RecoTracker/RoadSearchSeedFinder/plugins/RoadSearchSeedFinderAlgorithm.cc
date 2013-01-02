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
// $Author: eulisse $
// $Date: 2012/10/24 08:32:20 $
// $Revision: 1.1 $
//

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "RoadSearchSeedFinderAlgorithm.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetNew.h"
//***top-bottom
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//***
const double speedOfLight = 2.99792458e8;
const double unitCorrection = speedOfLight * 1e-2 * 1e-9;

RoadSearchSeedFinderAlgorithm::RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf) { 


  minPt_                      = conf.getParameter<double>("MinimalReconstructedTransverseMomentum");
  maxBarrelImpactParameter_   = conf.getParameter<double>("MaximalBarrelImpactParameter");
  maxEndcapImpactParameter_   = conf.getParameter<double>("MaximalEndcapImpactParameter");
  phiRangeDetIdLookup_        = conf.getParameter<double>("PhiRangeForDetIdLookupInRings");
  mergeSeedsCenterCut_A_      = conf.getParameter<double>("MergeSeedsCenterCut_A");
  mergeSeedsRadiusCut_A_      = conf.getParameter<double>("MergeSeedsRadiusCut_A");
  mergeSeedsCenterCut_B_      = conf.getParameter<double>("MergeSeedsCenterCut_B");
  mergeSeedsRadiusCut_B_      = conf.getParameter<double>("MergeSeedsRadiusCut_B");
  mergeSeedsCenterCut_C_      = conf.getParameter<double>("MergeSeedsCenterCut_C");
  mergeSeedsRadiusCut_C_      = conf.getParameter<double>("MergeSeedsRadiusCut_C");
  mergeSeedsCenterCut_        = mergeSeedsCenterCut_A_;
  mergeSeedsRadiusCut_        = mergeSeedsRadiusCut_A_;
  mergeSeedsDifferentHitsCut_ = conf.getParameter<unsigned int>("MergeSeedsDifferentHitsCut");
  mode_                       = conf.getParameter<std::string>("Mode");
  
  //special parameters for cosmic track reconstruction
  //cosmicTracking_             = conf.getParameter<bool>("CosmicTracking");
  //maxNumberOfClusters_        = conf.getParameter<unsigned int>("MaxNumberOfClusters");


  // safety check for mode
  if ( mode_ != "STANDARD" && mode_ != "STRAIGHT-LINE" ) {
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

  maxNumberOfSeeds_ = conf.getParameter<int32_t>("MaxNumberOfSeeds");
  //***top-bottom
  allPositiveOnly = conf.getParameter<bool>("AllPositiveOnly");
  allNegativeOnly = conf.getParameter<bool>("AllNegativeOnly");
  //***
}

RoadSearchSeedFinderAlgorithm::~RoadSearchSeedFinderAlgorithm() {
}


void RoadSearchSeedFinderAlgorithm::run(const SiStripRecHit2DCollection* rphiRecHits,
					const SiStripRecHit2DCollection* stereoRecHits,
					const SiStripMatchedRecHit2DCollection* matchedRecHits,
					const SiPixelRecHitCollection* pixelRecHits,
					const edm::EventSetup& es,
					RoadSearchSeedCollection &output)
{

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology* tTopo=tTopoHand.product();

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
  edm::ESHandle<MagneticField> magnetHandle;
  es.get<IdealMagneticFieldRecord>().get(magnetHandle);
  magnet_ = magnetHandle.product();

  //***top-bottom
  //TTRHBuilder
  TkTransientTrackingRecHitBuilder builder(tracker_,0,0,0,false);
  //***

  // get magnetic field for 0,0,0 , approximation for minRadius calculation
  beamSpotZMagneticField_ = magnet_->inTesla(GlobalPoint(0,0,0)).z();
  // calculate minimal radius at globalPoint in cm, take the z component of the magnetic field at GlobalPoint 2
  if ( beamSpotZMagneticField_ == 0 ) {
    minRadius_ = 999999999999.;
  } else {
    minRadius_ = minPt_ / 0.3 / beamSpotZMagneticField_ * 100;
  }

  // temporary storing collection of circle seeds
  std::vector<RoadSearchCircleSeed> localCircleSeeds;

  // loop over seed Ring pairs
  for ( Roads::const_iterator road = roads_->begin(); road != roads_->end(); ++road ) {

    localCircleSeeds.clear();

    const Roads::RoadSeed  *seed  = &((*road).first);
    const Roads::RoadSet   *set  =  &((*road).second);

    // determine seeding cuts from inner seed ring |eta|
    double r = std::abs((*seed->first.begin())->getrmax() + (*seed->first.begin())->getrmin())/2.;
    double z = std::abs((*seed->first.begin())->getzmax() + (*seed->first.begin())->getzmin())/2.;
    double eta = std::abs(std::log(std::tan(std::atan2(r,z)/2.)));

    if ( eta < 1.1 ) {
      mergeSeedsCenterCut_ = mergeSeedsCenterCut_A_;
      mergeSeedsRadiusCut_ = mergeSeedsRadiusCut_A_;
    } else if ( (eta >= 1.1) && (eta < 1.6) ) {
      mergeSeedsCenterCut_ = mergeSeedsCenterCut_B_;
      mergeSeedsRadiusCut_ = mergeSeedsRadiusCut_B_;
    } else if ( eta >= 1.6 ) {
      mergeSeedsCenterCut_ = mergeSeedsCenterCut_C_;
      mergeSeedsRadiusCut_ = mergeSeedsRadiusCut_C_;
    }

    if ( mode_ == "STRAIGHT-LINE" ) {

      // loop over seed ring pairs
      // draw straight line
      for ( std::vector<const Ring*>::const_iterator innerSeedRing = seed->first.begin();
	    innerSeedRing != seed->first.end();
	    ++innerSeedRing) {
	for ( std::vector<const Ring*>::const_iterator outerSeedRing = seed->second.begin();
	      outerSeedRing != seed->second.end();
	      ++outerSeedRing) {
	  calculateCircleSeedsFromRingsOneInnerOneOuter(localCircleSeeds,
							seed,
							set,
							*innerSeedRing,
							*outerSeedRing);
	  
	}
      }
    } else if ( mode_ == "STANDARD" ) {

      // take combinations of one inner and two outer or two inner and one outer seed ring
      for ( std::vector<const Ring*>::const_iterator innerSeedRing1 = seed->first.begin();
	    innerSeedRing1 != seed->first.end();
	    ++innerSeedRing1) {
	// two inner, one outer
	for ( std::vector<const Ring*>::const_iterator innerSeedRing2 = innerSeedRing1+1;
	      innerSeedRing2 != seed->first.end();
	      ++innerSeedRing2) {
	  if ( !ringsOnSameLayer(*innerSeedRing1,*innerSeedRing2,tTopo) ) {
	    for ( std::vector<const Ring*>::const_iterator outerSeedRing = seed->second.begin();
		  outerSeedRing != seed->second.end();
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
		calculateCircleSeedsFromRingsTwoInnerOneOuter(localCircleSeeds,
							      seed,
							      set,
							      *innerSeedRing1,
							      *innerSeedRing2,
							      *outerSeedRing);
	      }
	    }	  
	  }
	}
 	// one inner, two outer
 	for ( std::vector<const Ring*>::const_iterator outerSeedRing1 = seed->second.begin();
 	      outerSeedRing1 != seed->second.end();
 	      ++outerSeedRing1) {
 	  for ( std::vector<const Ring*>::const_iterator outerSeedRing2 = outerSeedRing1+1;
 		outerSeedRing2 != seed->second.end();
 		++outerSeedRing2) {
 	    if ( !ringsOnSameLayer(*outerSeedRing1,*outerSeedRing2,tTopo) ) {
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
 		calculateCircleSeedsFromRingsOneInnerTwoOuter(localCircleSeeds,
							      seed,
							      set,
 							      *innerSeedRing1,
 							      *outerSeedRing1,
 							      *outerSeedRing2);
 	      }
 	    }
 	  }
 	}
      }
    }

    // fill in eta mapped multi-map
    for ( std::vector<RoadSearchCircleSeed>::iterator localCircle = localCircleSeeds.begin(),
	    localCircleEnd = localCircleSeeds.end();
	  localCircle != localCircleEnd;
	  ++localCircle ) {
      RoadSearchSeed seed;
      seed.setSet(localCircle->getSet());
      seed.setSeed(localCircle->getSeed());
      //***top-bottom
      bool allPositive = true;
      bool allNegative = true;
      //***
      for (std::vector<const TrackingRecHit*>::const_iterator hit = localCircle->begin_hits();
	   hit != localCircle->end_hits();
	   ++hit ) {
	seed.addHit(*hit);
	//***top-bottom
	double seedY = builder.build(*hit)->globalPosition().y();
	if (seedY>0) allNegative = false;
	if (seedY<0) allPositive = false;
	//***
      }
      //***top-bottom
      //output.push_back(seed);
      if (allPositive && allPositiveOnly) output.push_back(seed);
      if (allNegative && allNegativeOnly) output.push_back(seed);
      if (!allPositiveOnly && !allNegativeOnly) output.push_back(seed);
      //***
    }

    if ((maxNumberOfSeeds_ > 0) && (output.size() > size_t(maxNumberOfSeeds_))) {
      edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
      output.clear(); 
      break;
    }
  }
    
  usedSeedRingCombinations_.clear();
  edm::LogInfo("RoadSearch") << "Found " << output.size() << " seeds."; 

}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromRingsTwoInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
										  const Roads::RoadSeed *seed,
										  const Roads::RoadSet *set,
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
	calculateCircleSeedsFromHits(circleSeeds, seed, set, ring1GlobalPoint, *ring1RecHit, ring2RecHits, ring3RecHits);
      }

    }
  }

  return result;

}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromRingsOneInnerTwoOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
										  const Roads::RoadSeed *seed,
										  const Roads::RoadSet *set,
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
	calculateCircleSeedsFromHits(circleSeeds, seed, set, ring1GlobalPoint, *ring1RecHit, ring2RecHits, ring3RecHits);
      }

    }
  }

  return result;

}
								       
bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromRingsOneInnerOneOuter(std::vector<RoadSearchCircleSeed> &circleSeeds,
										  const Roads::RoadSeed *seed,
										  const Roads::RoadSet *set,
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
	calculateCircleSeedsFromHits(circleSeeds, seed, set, ring1GlobalPoint, *ring1RecHit, ring2RecHits);
      }

    }
  }

  return result;

}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
								 const Roads::RoadSeed *seed,
								 const Roads::RoadSet *set,
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
            
      bool addCircle = false;
      if ( circle.Type() == RoadSearchCircleSeed::straightLine ) {
	addCircle = true;
      } else {
	if ( (circle.Radius() > minRadius_) &&
	     ((circle.InBarrel() &&
	       circle.ImpactParameter() < maxBarrelImpactParameter_) ||
	      (!circle.InBarrel() &&
	       circle.ImpactParameter() < maxEndcapImpactParameter_)) ) {
	  addCircle = true;
	
	  // check if circle compatible with previous circles, if not, add
	  for (std::vector<RoadSearchCircleSeed>::iterator alreadyContainedCircle = circleSeeds.begin(),
		 alreadyContainedCircleEnd = circleSeeds.end();
	       alreadyContainedCircle != alreadyContainedCircleEnd;
	       ++alreadyContainedCircle ) {
	    if ( circle.Compare(&*alreadyContainedCircle,
				mergeSeedsCenterCut_,
				mergeSeedsRadiusCut_,
				mergeSeedsDifferentHitsCut_) ) {
	      addCircle = false;
	      break;
	    }
	  }
	}
      }
      
      if ( addCircle ) {
	circle.setSeed(seed);
	circle.setSet(set);
	circleSeeds.push_back(circle);
      }
    }
  }

  return result;
}

bool RoadSearchSeedFinderAlgorithm::calculateCircleSeedsFromHits(std::vector<RoadSearchCircleSeed> &circleSeeds,
								 const Roads::RoadSeed *seed,
								 const Roads::RoadSet *set,
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
    circle.setSeed(seed);
    circle.setSet(set);
    circleSeeds.push_back(circle);
  }

  return result;
}

bool RoadSearchSeedFinderAlgorithm::ringsOnSameLayer(const Ring *ring1, const Ring* ring2, const TrackerTopology *tTopo) {
  //
  // check whether two input rings are on the same layer
  //

  // return value
  bool result = false;

  // get first DetId of ring
  const DetId ring1DetId = ring1->getFirst();
  const DetId ring2DetId = ring2->getFirst();

  result = detIdsOnSameLayer(ring1DetId,ring2DetId, tTopo);
  
  return result;
}

bool RoadSearchSeedFinderAlgorithm::detIdsOnSameLayer(DetId id1, DetId id2, const TrackerTopology *tTopo) {
  //
  // check whether two detids are on the same layer
  //

  if (id1.subdetId() == id2.subdetId() )
    if ( tTopo->layer(id1)==tTopo->layer(id2)) 
      return true;

  return false;
}


unsigned int RoadSearchSeedFinderAlgorithm::ClusterCounter(const edmNew::DetSetVector<SiStripCluster>* clusters) {

  const edmNew::DetSetVector<SiStripCluster>& input = *clusters;

  unsigned int totalClusters = 0;

  //loop over detectors
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(); DSViter!=input.end();DSViter++ ) {
    totalClusters+=DSViter->size();
  }

  return totalClusters;
}

//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudMakerAlgorithm
// 
// Description:     
//                  Road categories determined by outer Seed RecHit
//                  	RPhi: outer Seed RecHit in the Barrel
//                  	ZPhi: outer Seed RecHit in the Disks
//                  use inner and outer Seed RecHit and BeamSpot to calculate extrapolation
//                  	RPhi: phi = phi0 + asin(k r)
//                  	ZPhi: phi = phi0 + C z
//                  Loop over RoadSet, access Rings of Road
//                  	get average radius of Ring
//                  	use extrapolation to calculate phi_ref at average Ring radius
//                  	determine window in phi for DetId lookup in the Ring
//                  		phi_ref ± phi_window
//                  			PARAMETER: phi_window = pi/24
//                  	loop over DetId's in phi window in Ring
//                  		two cases (problem of phi = 0/2pi):
//                  			lower window border < upper window border
//                  			upper window border < lower window border
//                  Loop over RecHits of DetId
//                  	check compatibility of RecHit with extrapolation (delta phi cut)
//                  	single layer sensor treatment
//                  		RPhi:
//                  			stereo and barrel single layer sensor
//                  				calculate delta-phi
//                  			disk single layer sensor (r coordinate not well defined)
//                  				calculate delta phi between global positions of maximal strip extension and reference
//                  		ZPhi:
//                  			stereo sensor
//                  				calculate delta-phi
//                  			barrel single layer sensor (z coordinate not well defined)
//                  				calculate delta phi between global positions of maximal strip extension and reference
//                  			disk single layer sensor (tilted strips relative to local coordinate system of sensor
//                  				calculate delta phi between global positions of maximal strip extension and reference
//                  Check delta phi cut
//                  	cut value can be calculated based on extrapolation and Seed (pT dependent delta phi cut)
//                  	currently: constant delta phi cut (PARAMETER)
//                  		fill RecHit into Cloud
//                  			do not fill more than 32 RecHits per DetID into cloud (PARAMETER)
//                  first stage of Cloud cleaning cuts:
//                  	number of layers with hits in cloud (PARAMETER)
//                  	number of layers with no hits in cloud (PARAMETER)
//                  	number of consecutive layers with no hits in cloud (PARAMETER)
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/03/23 01:59:41 $
// $Revision: 1.9 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMakerAlgorithm.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"


#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

using namespace std;

double RoadSearchCloudMakerAlgorithm::epsilon      =   0.000000001;
double RoadSearchCloudMakerAlgorithm::half_pi      =   1.570796327;

RoadSearchCloudMakerAlgorithm::RoadSearchCloudMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

RoadSearchCloudMakerAlgorithm::~RoadSearchCloudMakerAlgorithm() {
}

double RoadSearchCloudMakerAlgorithm::map_phi(double phi) {
  // map phi to [0,2pi]
  double result = phi;
  const double pi = 3.14159265358979312;
  const double twopi = 2*pi;
  if ( result < -twopi ) result = result + twopi;
  if ( result < 0)       result = twopi + result;
  if ( result > twopi)   result = result - twopi;
  return result;
}

void RoadSearchCloudMakerAlgorithm::run(edm::Handle<TrajectorySeedCollection> input,
					const SiStripRecHit2DLocalPosCollection* rphiRecHits,
					const SiStripRecHit2DLocalPosCollection* stereoRecHits,
					const edm::EventSetup& es,
					RoadSearchCloudCollection &output)
{

  // get roads
  edm::ESHandle<Roads> roads;
  es.get<TrackerDigiGeometryRecord>().get(roads);

  // get trajectoryseed collection
  const TrajectorySeedCollection* inputSeeds = input.product();

  // array for layer information
  // information in vector of subdetector layer sizes always:
  // TIB, TOB, TID, TEC, PXB, PXF
  Roads::NumberOfLayersPerSubdetector numberOfLayersPerSubdetector = roads->getNumberOfLayersPerSubdetector();
  unsigned int totalNumberOfLayersPerSubdetector = 0;
  for ( Roads::NumberOfLayersPerSubdetectorConstIterator component = numberOfLayersPerSubdetector.begin(); component != numberOfLayersPerSubdetector.end(); ++component) {
    totalNumberOfLayersPerSubdetector += *component;
  }
  std::vector<bool> usedLayersArray(totalNumberOfLayersPerSubdetector);

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  // counter for seeds for edm::Ref size_type
  int seedCounter = -1;

  // loop over seeds
  for ( TrajectorySeedCollection::const_iterator seed = inputSeeds->begin(); seed != inputSeeds->end(); ++seed) {

    ++seedCounter;

    // get DetIds of SiStripRecHit2DLocalPos of Seed, assuming first is hit from inner SeedRing, second is hit from outer SeedRing
    if ( seed->nHits() < 2 ) {
      edm::LogError("RoadWarning") << "Seed has less then two linked TrackingRecHit, do not consider this seed.";
    } else {

      const TrackingRecHit *innerSeedRingHit = &(*(seed->recHits().first));
      const TrackingRecHit *outerSeedRingHit = &(*(--(seed->recHits().second)));

      // get RoadSeed from Roads
      const Roads::RoadSeed *roadSeed = roads->getRoadSeed(innerSeedRingHit->geographicalId(),outerSeedRingHit->geographicalId());
      const Roads::type roadType = roads->getRoadType(roadSeed);

      // get global positions of the hits
      GlobalPoint innerSeedHitGlobalPosition = tracker->idToDet(innerSeedRingHit->geographicalId())->surface().toGlobal(innerSeedRingHit->localPosition());
      GlobalPoint outerSeedHitGlobalPosition = tracker->idToDet(outerSeedRingHit->geographicalId())->surface().toGlobal(outerSeedRingHit->localPosition());

      // extrapolation parameters, phio: [0,2pi]
      double phi0 = -99.;
      double k0   = -99999999.99;

      // calculate phi0 and k0 dependent on RoadType
      if ( roadType == Roads::RPhi ) {
	double dr = outerSeedHitGlobalPosition.perp() - innerSeedHitGlobalPosition.perp();
	const double dr_min = 1; // cm
	if ( dr < dr_min ) {
	  edm::LogWarning("RoadSearch") << "RPhi road: seed Hits distance smaller than 1 cm, do not consider this seed.";
	} else {
	  // calculate r-phi extrapolation: phi = phi0 + asin(k0 r)
	  double det = innerSeedHitGlobalPosition.x() * outerSeedHitGlobalPosition.y() - innerSeedHitGlobalPosition.y() * outerSeedHitGlobalPosition.x();
	  if ( det == 0 ) {
	    edm::LogWarning("RoadSearch") << "RPhi road: 'det' == 0, do not consider this seed.";
	  } else {

	    // 	    double squaredGlobalRadiusInnerSeedHit = innerSeedHitGlobalPosition.x() * innerSeedHitGlobalPosition.x() +
	    // 	      innerSeedHitGlobalPosition.y() * innerSeedHitGlobalPosition.y();
	    // 	    double squaredGlobalRadiusOuterSeedHit = outerSeedHitGlobalPosition.x() * outerSeedHitGlobalPosition.x() +
	    // 	      outerSeedHitGlobalPosition.y() * outerSeedHitGlobalPosition.y();
	    // 	    double xc_det = squaredGlobalRadiusInnerSeedHit * outerSeedHitGlobalPosition.y() -
	    // 	      squaredGlobalRadiusOuterSeedHit * innerSeedHitGlobalPosition.y();
	    // 	    double yc_det = innerSeedHitGlobalPosition.x() * squaredGlobalRadiusOuterSeedHit -
	    // 	      outerSeedHitGlobalPosition.x() * squaredGlobalRadiusInnerSeedHit;

	    //srw	    k0 = det / sqrt(xc_det*xc_det + yc_det*yc_det);
	    //srw	    phi0 = map_phi(innerSeedHitGlobalPosition.phi() - std::asin(k0*innerSeedHitGlobalPosition.mag()));
            makecircle(innerSeedHitGlobalPosition.x(),innerSeedHitGlobalPosition.y(),
                       outerSeedHitGlobalPosition.x(),outerSeedHitGlobalPosition.y(),0.0,0.0);
            phi0 = phi0h;
            k0 = omegah;

	  }
	}
      } else {
	double dz = outerSeedHitGlobalPosition.z() - innerSeedHitGlobalPosition.z();
	const double dz_min = 1.e-6; // cm;
	if ( std::abs(dz) < dz_min ) {
	  edm::LogWarning("RoadSearch") << "ZPhi road: seed Hits are less than .01 microns away in z, do not consider this seed.";
	} else {
	  // calculate z-phi extrapolation: phi = phi0 + k0 z
	  k0 = map_phi(outerSeedHitGlobalPosition.phi() - innerSeedHitGlobalPosition.phi()) / dz;
	  phi0 =  map_phi(innerSeedHitGlobalPosition.phi() - k0 * innerSeedHitGlobalPosition.z());
	}
      }
      
      // continue if valid extrapolation parameters have been found
      if ( (phi0 != -99.) && (k0 != -99999999.99) ) {
	const Roads::RoadMapConstRange roadSets = roads->getRoadSet(roadSeed);

	for ( Roads::const_iterator roadMapEntry = roadSets.first; roadMapEntry != roadSets.second; ++roadMapEntry ) {

	  // create cloud
	  RoadSearchCloud cloud;

	  // seed edm::Ref
	  RoadSearchCloud::SeedRef seedRef(input,seedCounter);

	  cloud.addSeed(seedRef);

	  // reset array for layer information
	  for ( unsigned int layer = 0; layer < totalNumberOfLayersPerSubdetector; ++layer ) {
	    usedLayersArray[layer] = false;
	  }

	  for ( Roads::RoadSetConstIterator ring = roadMapEntry->second.begin(); ring != roadMapEntry->second.end(); ++ring ) {

	    // calculate phi-range for lookup of DetId's in Rings of RoadSet
	    // calculate phi at radius of Ring using extrapolation, Ring radius average of Ring.rmin() and Ring.rmax()
	    double ringRadius = ring->getrmin() + (ring->getrmax()-ring->getrmin())/2;
	    double ringZ      = ring->getzmin() + (ring->getzmax()-ring->getzmin())/2;
	    double ringPhi = 0.0;
	    if ( roadType == Roads::RPhi ) {
	      ringPhi = phiFromExtrapolation(phi0,k0,ringRadius,roadType);
	    } else {
	      ringPhi = phiFromExtrapolation(phi0,k0,ringZ,roadType);
	    }

	    // calculate range in phi around ringPhi
	    double ringHalfPhiExtension = conf_.getParameter<double>("MinimumHalfRoad");
	    double upperPhiRangeBorder = ringPhi + ringHalfPhiExtension;
	    double lowerPhiRangeBorder = ringPhi - ringHalfPhiExtension;

	    const double pi = 3.14159265358979312;

	    const std::vector<DetId> availableIDs = rphiRecHits->ids();
	    const std::vector<DetId> availableIDs2 = stereoRecHits->ids();

	    if ( lowerPhiRangeBorder <= upperPhiRangeBorder ) {
	      for ( Ring::const_iterator detid = ring->lower_bound(lowerPhiRangeBorder); detid != ring->upper_bound(upperPhiRangeBorder); ++detid) {
		if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detid->second) ) {
		  FillRecHitsIntoCloud(detid->second,rphiRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
		if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),detid->second) ) {
		  FillRecHitsIntoCloud(detid->second,stereoRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
	      }
	    } else {
	      for ( Ring::const_iterator detid = ring->lower_bound(lowerPhiRangeBorder); detid != ring->upper_bound(2*pi); ++detid) {
		if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detid->second) ) {
		  FillRecHitsIntoCloud(detid->second,rphiRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
		if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),detid->second) ) {
		  FillRecHitsIntoCloud(detid->second,stereoRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
	      }
	      for ( Ring::const_iterator detid = ring->lower_bound(0); detid != ring->upper_bound(upperPhiRangeBorder); ++detid) {
		if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detid->second) ) {
		  FillRecHitsIntoCloud(detid->second,rphiRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
		if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),detid->second) ) {
		  FillRecHitsIntoCloud(detid->second,stereoRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
	      }
	    }

	  }

	  if ( checkMinimalNumberOfUsedLayers(usedLayersArray) && 
	       checkMaximalNumberOfMissedLayers(usedLayersArray,roadMapEntry->second,numberOfLayersPerSubdetector) && 
	       checkMaximalNumberOfConsecutiveMissedLayers(usedLayersArray,roadMapEntry->second,numberOfLayersPerSubdetector) ) {
	    output.push_back(cloud);
	    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
	      if ( roadType == Roads::RPhi ){ 
		LogDebug("RoadSearch") << "This r-phi seed yields a cloud with " <<cloud.size() <<" hits";
	      } else { LogDebug("RoadSearch") << "This z-phi seed yields a cloud with "<<cloud.size() <<" hits";
	      }
	    }
	  } else {
	    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 1 ) {
	      if ( roadType == Roads::RPhi ){ 
		LogDebug("RoadSearch") << "This r-phi seed yields no clouds";
	      } else { LogDebug("RoadSearch") << "This z-phi seed yields no clouds";
	      }
	    }
	  }
	}
      }
    }
  }

  edm::LogInfo("RoadSearch") << "Found " << output.size() << " clouds."; 

};

void RoadSearchCloudMakerAlgorithm::FillRecHitsIntoCloud(DetId id, const SiStripRecHit2DLocalPosCollection* inputRecHits, 
							 double phi0, double k0, Roads::type roadType, double ringPhi,
							 const TrajectorySeed* seed, std::vector<bool> &usedLayersArray, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector,
							 const TrackerGeometry *tracker, RoadSearchCloud &cloud) {
  // retrieve vector<SiStripRecHit2DLocalPos> for id, loop over SiStripRecHit2DLocalPos, check if compatible with cloud, fill into cloud

  const SiStripRecHit2DLocalPosCollection::range recHitRange = inputRecHits->get(id);

  for ( SiStripRecHit2DLocalPosCollection::const_iterator recHitIterator = recHitRange.first; recHitIterator != recHitRange.second; ++recHitIterator) {
    const SiStripRecHit2DLocalPos * recHit = &(*recHitIterator);

    unsigned int maxDetHitsInCloudPerDetId = conf_.getParameter<int>("MaxDetHitsInCloudPerDetId");

    if ( roadType == Roads::RPhi ) {
      if ( isSingleLayer(id) ) {
	if ( isBarrelSensor(id) ) {
	  //
	  //  This is where the barrel rphiRecHits end up for Roads::RPhi
	  //
          GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
	  double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
          double hitphi = map_phi(ghit.phi());
	  double phi = phiFromExtrapolation(phi0,k0,hitRadius,roadType);
	  //	  if ( std::abs(map_phi(phi-ringPhi)) < phiMax(seed,phi0,k0) ) {
	  //	  if ( std::abs(phi-ringPhi) < phiMax(seed,phi0,k0) ) {
	  if ( std::abs(hitphi-phi) < phiMax(seed,phi0,k0) ) {
	    if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
	      cloud.addHit((TrackingRecHit*)recHit->clone());
	      //next line was missing - added stevew feb-9-2006
              setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	    }
	  }
	} else {
	  LocalPoint hit = recHit->localPosition();
	  const TrapezoidalStripTopology *topology = dynamic_cast<const TrapezoidalStripTopology*>(&(tracker->idToDetUnit(id)->topology()));
	  double stripAngle = topology->stripAngle(topology->strip(hit));
	  double stripLength = topology->localStripLength(hit);
	  LocalPoint upperLocalBoundary(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	  LocalPoint lowerLocalBoundary(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	  double upperBoundaryRadius = tracker->idToDetUnit(id)->surface().toGlobal(upperLocalBoundary).perp(); 
	  double lowerBoundaryRadius = tracker->idToDetUnit(id)->surface().toGlobal(lowerLocalBoundary).perp();
	  double upperBoundaryPhi = phiFromExtrapolation(phi0,k0,upperBoundaryRadius,roadType);
	  double lowerBoundaryPhi = phiFromExtrapolation(phi0,k0,lowerBoundaryRadius,roadType);
	  double hitPhi = map_phi(tracker->idToDetUnit(id)->surface().toGlobal(hit).phi());

	  if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
	    //
	    //  This is where the disk (???) rphiRecHits end up for Roads::RPhi
	    //
	    if ( ((lowerBoundaryPhi - phiMax(seed,phi0,k0)) < hitPhi) &&
		 ((upperBoundaryPhi + phiMax(seed,phi0,k0)) > hitPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  } else {
	    //
	    //  some type of hit (see above) gets here
	    //
	    if ( ((upperBoundaryPhi - phiMax(seed,phi0,k0)) < hitPhi) &&
		 ((lowerBoundaryPhi + phiMax(seed,phi0,k0)) > hitPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  }
	}
      } else {
	//
	//  This is where the barrel stereoRecHits end up for Roads::RPhi
	//
        GlobalPoint ghit = tracker->idToDetUnit(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
	double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
        double hitphi = map_phi(ghit.phi());
	double phi = phiFromExtrapolation(phi0,k0,hitRadius,roadType);
	//	if ( std::abs(map_phi(phi-ringPhi)) < phiMax(seed,phi0,k0) ) {
	//	if ( std::abs(phi-ringPhi) < phiMax(seed,phi0,k0) ) {
        if ( std::abs(hitphi-phi) < 6.0*phiMax(seed,phi0,k0) ) {
	  if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
	    cloud.addHit((TrackingRecHit*)recHit->clone());
	    setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	  }
	}
      }
    } else {
      //
      // roadType == Roads::ZPhi
      //
      if ( isSingleLayer(id) ) {
	if ( isBarrelSensor(id) ) {
	  LocalPoint hit = recHit->localPosition();
	  const RectangularStripTopology *topology = dynamic_cast<const RectangularStripTopology*>(&(tracker->idToDetUnit(id)->topology()));
	  double stripLength = topology->stripLength();
	  LocalPoint upperLocalBoundary(hit.x(),hit.y()+stripLength/2,0);
	  LocalPoint lowerLocalBoundary(hit.x(),hit.y()-stripLength/2,0);
	  double upperBoundaryZ = tracker->idToDetUnit(id)->surface().toGlobal(upperLocalBoundary).z(); 
	  double lowerBoundaryZ = tracker->idToDetUnit(id)->surface().toGlobal(lowerLocalBoundary).z();
	  double upperBoundaryPhi = phiFromExtrapolation(phi0,k0,upperBoundaryZ,roadType);
	  double lowerBoundaryPhi = phiFromExtrapolation(phi0,k0,lowerBoundaryZ,roadType);
	  double hitPhi = map_phi(tracker->idToDetUnit(id)->surface().toGlobal(recHit->localPosition()).phi());

	  if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
	    //
	    //  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
	    //
	    if ( ((lowerBoundaryPhi - phiMax(seed,phi0,k0)) < hitPhi) &&
		 ((upperBoundaryPhi + phiMax(seed,phi0,k0)) > hitPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  } else {
	    //
	    //  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
	    //
	    if ( ((upperBoundaryPhi - phiMax(seed,phi0,k0)) < hitPhi) &&
		 ((lowerBoundaryPhi + phiMax(seed,phi0,k0)) > hitPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  }
	} else {
	  LocalPoint hit = recHit->localPosition();
	  const TrapezoidalStripTopology *topology = dynamic_cast<const TrapezoidalStripTopology*>(&(tracker->idToDetUnit(id)->topology()));
	  double stripAngle = topology->stripAngle(topology->strip(hit));
	  double stripLength = topology->localStripLength(hit);
	  LocalPoint upperLocalBoundary(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	  LocalPoint lowerLocalBoundary(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	  double upperBoundaryZ = tracker->idToDetUnit(id)->surface().toGlobal(upperLocalBoundary).z(); 
	  double lowerBoundaryZ = tracker->idToDetUnit(id)->surface().toGlobal(lowerLocalBoundary).z();
	  double upperBoundaryPhi = phiFromExtrapolation(phi0,k0,upperBoundaryZ,roadType);
	  double lowerBoundaryPhi = phiFromExtrapolation(phi0,k0,lowerBoundaryZ,roadType);

	  if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
	    //
	    //  This is where the disk rphiRecHits end up for Roads::ZPhi
	    //
	    if ( ((lowerBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((upperBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  } else {
	    //
	    //  no hits (see above) seem to get here
	    //
	    if ( ((upperBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((lowerBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  }
	}
      } else {
	double hitphi = map_phi(tracker->idToDetUnit(id)->surface().toGlobal(recHit->localPosition()).phi());
	//double hitRadius = tracker->idToDetUnit(id)->surface().toGlobal(recHit->localPosition()).perp();
	double hitZ = tracker->idToDetUnit(id)->surface().toGlobal(recHit->localPosition()).z();
	double phi = phiFromExtrapolation(phi0,k0,hitZ,roadType);
	//
	//  This is where the disk stereoRecHits end up for Roads::ZPhi
	//
	//	if ( std::abs(map_phi(phi-ringPhi)) < phiMax(seed,phi0,k0) ) {
	//	if ( std::abs(phi-ringPhi) < phiMax(seed,phi0,k0) ) {
        if ( std::abs(hitphi-phi) < 6.0*phiMax(seed,phi0,k0) ) {
	  if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
	    cloud.addHit((TrackingRecHit*)recHit->clone());
	    setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	  }
	}
      }
    }    
  }

}

bool RoadSearchCloudMakerAlgorithm::isSingleLayer(DetId id) {

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    if ( !tibid.stereo() ) {
      return true;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    if ( !tobid.stereo() ) {
      return true;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    if ( !tidid.stereo() ) {
      return true;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    if ( !tecid.stereo() ) {
      return true;
    }
  } else {
    return false;
  }

  return false;
}

bool RoadSearchCloudMakerAlgorithm::isBarrelSensor(DetId id) {

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    return true;
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    return true;
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    return true;
  } else {
    return false;
  }

}

double RoadSearchCloudMakerAlgorithm::phiFromExtrapolation(double phi0, double k0, double ringRadius, Roads::type roadType) {

  double ringPhi = -99.;
  if ( roadType == Roads::RPhi ) {
    //    ringPhi = map_phi(phi0 + std::asin ( k0 * ringRadius ));
    double omega=k0, d0=0.0, rl=ringRadius;
    double sp0=sin(phi0); double cp0=cos(phi0);  
    double xc=-sp0*(d0+1.0/omega);    
    double yc=cp0*(d0+1.0/omega);
    double rh=fabs(1.0/omega);
    double bbb=fabs(d0+1.0/omega);
    double sss=0.5*(rl+rh+bbb);
    double ddd=sqrt((sss-bbb)*(sss-rh)/(sss*(sss-rl)));
    double phil1=2.0*atan(ddd);
    double phit=phi0+phil1; if (omega<0.0)phit=phi0-phil1;
    double xh=xc+sin(phit)/omega;
    double yh=yc-cos(phit)/omega;
    double phih=atan2(yh,xh);
    ringPhi = map_phi(phih);
  } else {
    ringPhi = map_phi(phi0 + k0 * ringRadius);
  }

  return ringPhi;
}

double RoadSearchCloudMakerAlgorithm::phiMax(const TrajectorySeed *seed, double phi0, double k0) {

  return conf_.getParameter<double>("PhiRoadSize");

}

void RoadSearchCloudMakerAlgorithm::setLayerNumberArray(DetId id, std::vector<bool> &usedLayersArray, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector) {
  // order always: TIB, TOB, TID, TEC, PXB, PXF

  unsigned int index = getIndexInUsedLayersArray(id,numberOfLayersPerSubdetector);
  if ( (index != 999999) && (index <= usedLayersArray.size()) ) {
    usedLayersArray[index-1] = true;
  } else {
    edm::LogWarning("RoadSearch") << "SetLayerNumberArray couldn't set array entry for unknown Subdetector Component of DetId: " << id.rawId();
  }
}

unsigned int RoadSearchCloudMakerAlgorithm::getIndexInUsedLayersArray(DetId id, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector) {

  unsigned int index = 0;
  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    index = tibid.layer();
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    index = (tobid.layer()+numberOfLayersPerSubdetector[0]);
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    // fill negative side first
    if ( tidid.side() == 1 ) {
      index = (tidid.wheel()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]);
    } else {
      // assume only even total number of wheels
      index = (tidid.wheel()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]+numberOfLayersPerSubdetector[2]/2);
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    // fill negative side first
    if ( tecid.side() == 1 ) {
      index = (tecid.wheel()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]+numberOfLayersPerSubdetector[2]);
    } else {
      // assume only even total number of wheels
      index = (tecid.wheel()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]+numberOfLayersPerSubdetector[2]+numberOfLayersPerSubdetector[3]/2);
    }
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    PXBDetId pxbid(id.rawId()); 
    index = (pxbid.layer()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]+numberOfLayersPerSubdetector[2]+numberOfLayersPerSubdetector[3]);
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
    PXFDetId pxfid(id.rawId());
    // fill negative side first
    if ( pxfid.side() == 1 ) {
      index = (pxfid.disk()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]+numberOfLayersPerSubdetector[2]+numberOfLayersPerSubdetector[3]+numberOfLayersPerSubdetector[4]);
    } else {
      // assume only even total number of wheels
      index = (pxfid.disk()+numberOfLayersPerSubdetector[0]+numberOfLayersPerSubdetector[1]+numberOfLayersPerSubdetector[2]+numberOfLayersPerSubdetector[3]+numberOfLayersPerSubdetector[4]+numberOfLayersPerSubdetector[5]/2);
    }
  } else {
    edm::LogWarning("RoadSearch") << "GetIndexInUsedLayersArray couldn't return array entry for unknown Subdetector Component of DetId: " << id.rawId();
    return 999999;
  }

  return index;

}



bool RoadSearchCloudMakerAlgorithm::checkMinimalNumberOfUsedLayers(std::vector<bool> &usedLayersArray){

  bool result = false;
  
  unsigned int numberOfUsedLayers = 0;

  for (std::vector<bool>::const_iterator layer = usedLayersArray.begin(); layer != usedLayersArray.end(); ++layer) {
    if ( *layer == true ) {
      ++numberOfUsedLayers; 
    }
  }

  if ( numberOfUsedLayers >= (unsigned int)conf_.getParameter<int>("MinimalNumberOfUsedLayersPerRoad") ) {
    result = true;
  }
  if (!result && conf_.getUntrackedParameter<int>("VerbosityLevel") > 1) 
    edm::LogWarning("RoadSearch") << " Cloud only has "<<numberOfUsedLayers<<" hits";
  return result;

}

bool RoadSearchCloudMakerAlgorithm::checkMaximalNumberOfMissedLayers(std::vector<bool> &usedLayersArray, const Roads::RoadSet &roadSet, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector){

  bool result = false;

  unsigned int missed = 0;

  for ( Roads::RoadSetConstIterator ring = roadSet.begin(); ring != roadSet.end(); ++ring) {
    unsigned int index = getIndexInUsedLayersArray(ring->getFirst(),numberOfLayersPerSubdetector);
    if ( (index != 999999) && (index < usedLayersArray.size()) ) {
      if ( usedLayersArray[index] == false ) {
	++missed;
      }
    }
  }

  if ( missed <= (unsigned int)conf_.getParameter<int>("MaximalNumberOfMissedLayersPerRoad") ) {
    result = true;
  }

  if (!result){
    result = true;
  }

  return result;

}

bool RoadSearchCloudMakerAlgorithm::checkMaximalNumberOfConsecutiveMissedLayers(std::vector<bool> &usedLayersArray, const Roads::RoadSet &roadSet, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector){

  bool result = false;

  Roads::RoadSet sorted = roadSet;
  sort(sorted.begin(),sorted.end());

  unsigned int missed = 0;
  unsigned int temp_missed = 0;

  for ( Roads::RoadSetConstIterator ring = roadSet.begin(); ring != roadSet.end(); ++ring) {
    unsigned int index = getIndexInUsedLayersArray(ring->getFirst(),numberOfLayersPerSubdetector);
    if ( (index != 999999) && (index < usedLayersArray.size()) ) {
      if ( usedLayersArray[index] == false ) {
	++temp_missed;
      } else {
	if ( temp_missed > missed ) {
	  missed = temp_missed;
	}
	temp_missed = 0;
      }
    }
  }  
  //never updated if all missed layers were at end of list
  if ( temp_missed > missed ) {missed = temp_missed;}

  if ( missed <= (unsigned int)conf_.getParameter<int>("MaximalNumberOfConsecutiveMissedLayersPerRoad") ) {
    result = true;
  }

  if (!result){
    result = true;
  }

  return result;

}

void RoadSearchCloudMakerAlgorithm::makecircle(double x1, double y1, 
					       double x2,double y2, double x3, double y3){
  //    cout << "x1 y1 " << x1 << " " << y1 << endl;
  //    cout << "x2 y2 " << x2 << " " << y2 << endl;
  //    cout << "x3 y3 " << x3 << " " << y3 << endl;
  double x1t=x1-x3; double y1t=y1-y3; double r1s=x1t*x1t+y1t*y1t;
  double x2t=x2-x3; double y2t=y2-y3; double r2s=x2t*x2t+y2t*y2t;
  double rho=x1t*y2t-x2t*y1t;
  double xc, yc, rc, fac;
  if (fabs(rho)<RoadSearchCloudMakerAlgorithm::epsilon){
    rc=1.0/(RoadSearchCloudMakerAlgorithm::epsilon);
    fac=sqrt(x1t*x1t+y1t*y1t);
    xc=x2+y1t*rc/fac;
    yc=y2-x1t*rc/fac;
  }else{
    fac=0.5/rho;
    xc=fac*(r1s*y2t-r2s*y1t);
    yc=fac*(r2s*x1t-r1s*x2t); 
    rc=sqrt(xc*xc+yc*yc); xc+=x3; yc+=y3;
  }
  double s3=0.0;
  double f1=x1*yc-y1*xc; double f2=x2*yc-y2*xc; 
  double f3=x3*yc-y3*xc;
  if ((f1<0.0)&&(f2<0.0)&&(f3<=0.0))s3=1.0;
  if ((f1>0.0)&&(f2>0.0)&&(f3>=0.0))s3=-1.0;
  d0h=-s3*(sqrt(xc*xc+yc*yc)-rc);
  phi0h=atan2(yc,xc)+s3*half_pi;
  omegah=-s3/rc;
}


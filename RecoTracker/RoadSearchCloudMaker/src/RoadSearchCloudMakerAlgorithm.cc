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
// $Author: stevew $
// $Date: 2006/02/13 19:32:28 $
// $Revision: 1.4 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMakerAlgorithm.h"

#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseSiStripRecHit2DLocalPos.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

using namespace std;

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

void RoadSearchCloudMakerAlgorithm::run(const TrackingSeedCollection* input,
			      const SiStripRecHit2DLocalPosCollection* rphiRecHits,
			      const SiStripRecHit2DLocalPosCollection* stereoRecHits,
			      const edm::EventSetup& es,
			      RoadSearchCloudCollection &output)
{

  // get roads
  edm::ESHandle<Roads> roads;
  es.get<TrackerDigiGeometryRecord>().get(roads);

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
  edm::ESHandle<TrackingGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  // loop over seeds
  for ( TrackingSeedCollection::const_iterator seed = input->begin(); seed != input->end(); ++seed) {

    // get DetIds of SiStripRecHit2DLocalPos of Seed, assuming first is hit from inner SeedRing, second is hit from outer SeedRing
    if ( seed->size() < 2 ) {
      std::cout << "[RoadSearchCloudMaker]: seed has less then two linked SiStripRecHit2DLocalPos, do not consider this seed." << std::endl;
    } else {

      const BaseSiStripRecHit2DLocalPos *innerSeedRingHit = *(seed->begin());
      const BaseSiStripRecHit2DLocalPos *outerSeedRingHit = *(--seed->end());

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
	double dr = outerSeedHitGlobalPosition.mag() - innerSeedHitGlobalPosition.mag();
	const double dr_min = 1; // cm
	if ( dr < dr_min ) {
	  std::cout << "[RoadSearchCloudMaker]: RPhi road: seed Hits distance smaller than 1 cm, do not consider this seed." << std::endl;
	} else {
	  // calculate r-phi extrapolation: phi = phi0 + asin(k0 r)
	  double det = innerSeedHitGlobalPosition.x() * outerSeedHitGlobalPosition.y() - innerSeedHitGlobalPosition.y() * outerSeedHitGlobalPosition.x();
	  if ( det == 0 ) {
	    std::cout << "[RoadSearchCloudMaker]: RPhi road: 'det' == 0, do not consider this seed." << std::endl;
	  } else {

	    double squaredGlobalRadiusInnerSeedHit = innerSeedHitGlobalPosition.x() * innerSeedHitGlobalPosition.x() +
	      innerSeedHitGlobalPosition.y() * innerSeedHitGlobalPosition.y();
	    double squaredGlobalRadiusOuterSeedHit = outerSeedHitGlobalPosition.x() * outerSeedHitGlobalPosition.x() +
	      outerSeedHitGlobalPosition.y() * outerSeedHitGlobalPosition.y();
	    double xc_det = squaredGlobalRadiusInnerSeedHit * outerSeedHitGlobalPosition.y() -
	      squaredGlobalRadiusOuterSeedHit * innerSeedHitGlobalPosition.y();
	    double yc_det = innerSeedHitGlobalPosition.x() * squaredGlobalRadiusOuterSeedHit -
	      outerSeedHitGlobalPosition.x() * squaredGlobalRadiusInnerSeedHit;

	    k0 = det / sqrt(xc_det*xc_det + yc_det*yc_det);
	    phi0 = map_phi(innerSeedHitGlobalPosition.phi() - std::asin(k0*innerSeedHitGlobalPosition.mag()));
	  }
	}
      } else {
	double dz = outerSeedHitGlobalPosition.z() - innerSeedHitGlobalPosition.z();
	const double dz_min = 1.e-6; // cm;
	if ( std::abs(dz) < dz_min ) {
	  std::cout << "[RoadSearchCloudMaker]: ZPhi road: seed Hits are less than .01 microns away in z, do not consider this seed." << std::endl;
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

	  // reset array for layer information
	  for ( unsigned int layer = 0; layer < totalNumberOfLayersPerSubdetector; ++layer ) {
	    usedLayersArray[layer] = false;
	  }

	  for ( Roads::RoadSetConstIterator ring = roadMapEntry->second.begin(); ring != roadMapEntry->second.end(); ++ring ) {

	    // calculate phi-range for lookup of DetId's in Rings of RoadSet
	    // calculate phi at radius of Ring using extrapolation, Ring radius average of Ring.rmin() and Ring.rmax()
	    double ringRadius = ring->getrmin() + (ring->getrmax()-ring->getrmin())/2;
	    double ringPhi = phiFromExtrapolation(phi0,k0,ringRadius,roadType);

	    // calculate range in phi around ringPhi
	    double ringHalfPhiExtension = conf_.getParameter<double>("MinimumHalfRoad");
	    double upperPhiRangeBorder = ringPhi + ringHalfPhiExtension;
	    double lowerPhiRangeBorder = ringPhi - ringHalfPhiExtension;

	    const double pi = 3.14159265358979312;

	    const std::vector<unsigned int> availableIDs = rphiRecHits->detIDs();
	    const std::vector<unsigned int> availableIDs2 = stereoRecHits->detIDs();

	    if ( lowerPhiRangeBorder <= upperPhiRangeBorder ) {
	      for ( Ring::const_iterator detid = ring->lower_bound(lowerPhiRangeBorder); detid != ring->upper_bound(upperPhiRangeBorder); ++detid) {
		if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detid->second.rawId()) ) {
		  FillRecHitsIntoCloud(detid->second,rphiRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
		if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),detid->second.rawId()) ) {
		  FillRecHitsIntoCloud(detid->second,stereoRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
	      }
	    } else {
	      for ( Ring::const_iterator detid = ring->lower_bound(lowerPhiRangeBorder); detid != ring->upper_bound(2*pi); ++detid) {
		if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detid->second.rawId()) ) {
		  FillRecHitsIntoCloud(detid->second,rphiRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
		if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),detid->second.rawId()) ) {
		  FillRecHitsIntoCloud(detid->second,stereoRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
	      }
	      for ( Ring::const_iterator detid = ring->lower_bound(0); detid != ring->upper_bound(upperPhiRangeBorder); ++detid) {
		if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detid->second.rawId()) ) {
		  FillRecHitsIntoCloud(detid->second,rphiRecHits,phi0,k0,roadType,ringPhi,&(*seed),
                                       usedLayersArray,numberOfLayersPerSubdetector,tracker.product(),cloud);
		}
		if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),detid->second.rawId()) ) {
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
	  }
	}
      }
    }
  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << "[RoadSearchCloudMakerAlgorithm] found " << output.size() << " clouds." << endl; 
  }

};

void RoadSearchCloudMakerAlgorithm::FillRecHitsIntoCloud(DetId id, const SiStripRecHit2DLocalPosCollection* inputRecHits, 
					       double phi0, double k0, Roads::type roadType, double ringPhi,
					       const TrackingSeed* seed, std::vector<bool> &usedLayersArray, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector,
					       const TrackingGeometry *tracker, RoadSearchCloud &cloud) {
  // retrieve vector<SiStripRecHit2DLocalPos> for id, loop over SiStripRecHit2DLocalPos, check if compatible with cloud, fill into cloud

  const SiStripRecHit2DLocalPosCollection::Range recHitRange = inputRecHits->get(id.rawId());

  for ( SiStripRecHit2DLocalPosCollection::ContainerConstIterator recHitIterator = recHitRange.first; recHitIterator != recHitRange.second; ++recHitIterator) {
    SiStripRecHit2DLocalPos *recHit = &(*recHitIterator);

    unsigned int maxDetHitsInCloudPerDetId = conf_.getParameter<int>("MaxDetHitsInCloudPerDetId");

    if ( roadType == Roads::RPhi ) {
      if ( isSingleLayer(id) ) {
	if ( isBarrelSensor(id) ) {
//
//  This is where the barrel rphiRecHits end up for Roads::RPhi
//
	  double hitRadius = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition()).mag();
	  double phi = phiFromExtrapolation(phi0,k0,hitRadius,roadType);
//	  if ( std::abs(map_phi(phi-ringPhi)) < phiMax(seed,phi0,k0) ) {
	  if ( std::abs(phi-ringPhi) < phiMax(seed,phi0,k0) ) {
	    if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
	      cloud.addHit(recHit);
//next line was missing - added stevew feb-9-2006
              setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	    }
	  }
	} else {
	  LocalPoint hit = recHit->localPosition();
	  const TrapezoidalStripTopology *topology = dynamic_cast<const TrapezoidalStripTopology*>(&(tracker->idToDet(id)->topology()));
	  double stripAngle = topology->stripAngle(topology->strip(hit));
	  double stripLength = topology->localStripLength(hit);
	  LocalPoint upperLocalBoundary(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	  LocalPoint lowerLocalBoundary(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	  double upperBoundaryRadius = tracker->idToDet(id)->surface().toGlobal(upperLocalBoundary).mag(); 
	  double lowerBoundaryRadius = tracker->idToDet(id)->surface().toGlobal(lowerLocalBoundary).mag();
	  double upperBoundaryPhi = phiFromExtrapolation(phi0,k0,upperBoundaryRadius,roadType);
	  double lowerBoundaryPhi = phiFromExtrapolation(phi0,k0,lowerBoundaryRadius,roadType);

	  if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
//
//  This is where the disk (???) rphiRecHits end up for Roads::RPhi
//
	    if ( ((lowerBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((upperBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit(recHit);
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  } else {
//
//  some type of hit (see above) gets here
//
	    if ( ((upperBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((lowerBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit(recHit);
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  }
	}
      } else {
//
//  This is where the barrel stereoRecHits end up for Roads::RPhi
//
	double hitRadius = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition()).mag();
	double phi = phiFromExtrapolation(phi0,k0,hitRadius,roadType);
//	if ( std::abs(map_phi(phi-ringPhi)) < phiMax(seed,phi0,k0) ) {
	if ( std::abs(phi-ringPhi) < phiMax(seed,phi0,k0) ) {
	  if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
	    cloud.addHit(recHit);
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
	  const RectangularStripTopology *topology = dynamic_cast<const RectangularStripTopology*>(&(tracker->idToDet(id)->topology()));
	  double stripLength = topology->stripLength();
	  LocalPoint upperLocalBoundary(hit.x(),hit.y()+stripLength/2,0);
	  LocalPoint lowerLocalBoundary(hit.x(),hit.y()-stripLength/2,0);
	  double upperBoundaryRadius = tracker->idToDet(id)->surface().toGlobal(upperLocalBoundary).z(); 
	  double lowerBoundaryRadius = tracker->idToDet(id)->surface().toGlobal(lowerLocalBoundary).z();
	  double upperBoundaryPhi = phiFromExtrapolation(phi0,k0,upperBoundaryRadius,roadType);
	  double lowerBoundaryPhi = phiFromExtrapolation(phi0,k0,lowerBoundaryRadius,roadType);
	  
	  if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
//
//  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
//
	    if ( ((lowerBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((upperBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit(recHit);
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  } else {
//
//  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
//
	    if ( ((upperBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((lowerBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit(recHit);
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  }
	} else {
	  LocalPoint hit = recHit->localPosition();
	  const TrapezoidalStripTopology *topology = dynamic_cast<const TrapezoidalStripTopology*>(&(tracker->idToDet(id)->topology()));
	  double stripAngle = topology->stripAngle(topology->strip(hit));
	  double stripLength = topology->localStripLength(hit);
	  LocalPoint upperLocalBoundary(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	  LocalPoint lowerLocalBoundary(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	  double upperBoundaryRadius = tracker->idToDet(id)->surface().toGlobal(upperLocalBoundary).z(); 
	  double lowerBoundaryRadius = tracker->idToDet(id)->surface().toGlobal(lowerLocalBoundary).z();
	  double upperBoundaryPhi = phiFromExtrapolation(phi0,k0,upperBoundaryRadius,roadType);
	  double lowerBoundaryPhi = phiFromExtrapolation(phi0,k0,lowerBoundaryRadius,roadType);
	  
	  if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
//
//  This is where the disk rphiRecHits end up for Roads::ZPhi
//
	    if ( ((lowerBoundaryPhi - phiMax(seed,phi0,k0)) < ringPhi) &&
		 ((upperBoundaryPhi + phiMax(seed,phi0,k0)) > ringPhi) ) {
	      if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
		cloud.addHit(recHit);
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
		cloud.addHit(recHit);
		setLayerNumberArray(id,usedLayersArray,numberOfLayersPerSubdetector);
	      }
	    }
	  }
	}
      } else {
	double hitRadius = tracker->idToDet(id)->surface().toGlobal(recHit->localPosition()).mag();
	double phi = phiFromExtrapolation(phi0,k0,hitRadius,roadType);
//
//  This is where the disk stereoRecHits end up for Roads::ZPhi
//
//	if ( std::abs(map_phi(phi-ringPhi)) < phiMax(seed,phi0,k0) ) {
	if ( std::abs(phi-ringPhi) < phiMax(seed,phi0,k0) ) {
	  if ( cloud.size() < maxDetHitsInCloudPerDetId ) {
	    cloud.addHit(recHit);
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
    ringPhi = map_phi(phi0 + std::asin ( k0 * ringRadius ));
  } else {
    ringPhi = map_phi(phi0 + k0 * ringRadius);
  }

  return ringPhi;
}

double RoadSearchCloudMakerAlgorithm::phiMax(const TrackingSeed *seed, double phi0, double k0) {

  return conf_.getParameter<double>("PhiRoadSize");

}

void RoadSearchCloudMakerAlgorithm::setLayerNumberArray(DetId id, std::vector<bool> &usedLayersArray, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector) {
  // order always: TIB, TOB, TID, TEC, PXB, PXF

  unsigned int index = getIndexInUsedLayersArray(id,numberOfLayersPerSubdetector);
  if ( (index != 999999) && (index <= usedLayersArray.size()) ) {
    usedLayersArray[index-1] = true;
  } else {
    std::cout << "[RoadSearchCloudMakerAlgorithm]: setLayerNumberArray couldn't set array entry for unknown Subdetector Component of DetId: " << id.rawId() << std::endl;
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
    std::cout << "[RoadSearchCloudMakerAlgorithm]: getIndexInUsedLayersArray couldn't return array entry for unknown Subdetector Component of DetId: " << id.rawId() << std::endl;
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
//  std::cout << "Failed check forced true in RoadSearchCloudMakerAlgorithm::checkMaximalNumberOfMissedLayers" << endl;
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
//  std::cout << "Failed check forced true in RoadSearchCloudMakerAlgorithm::checkMaximalNumberOfConsecutiveMissedLayers" << endl;
  }

  return result;

}


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
//                 			barrel single layer sensor (z coordinate not well defined)
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
// $Author: innocent $
// $Date: 2012/01/17 15:41:12 $
// $Revision: 1.59 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RoadSearchCloudMakerAlgorithm.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
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
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchDetIdHelper.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

using namespace std;

double RoadSearchCloudMakerAlgorithm::epsilon      =   0.000000001;

RoadSearchCloudMakerAlgorithm::RoadSearchCloudMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
  recHitVectorClass.setMode(DetHitAccess::standard);    
  recHitVectorClass.use_rphiRecHits(conf_.getParameter<bool>("UseRphiRecHits"));
  recHitVectorClass.use_stereoRecHits(conf_.getParameter<bool>("UseStereoRecHits"));
  
  
  theRPhiRoadSize =  conf_.getParameter<double>("RPhiRoadSize");
  theZPhiRoadSize =  conf_.getParameter<double>("ZPhiRoadSize");
  UsePixels = conf_.getParameter<bool>("UsePixelsinRS");
  NoFieldCosmic = conf_.getParameter<bool>("StraightLineNoBeamSpotCloud");
  theMinimumHalfRoad = conf_.getParameter<double>("MinimumHalfRoad");
  
  maxDetHitsInCloudPerDetId = conf_.getParameter<unsigned int>("MaxDetHitsInCloudPerDetId");
  minFractionOfUsedLayersPerCloud = conf_.getParameter<double>("MinimalFractionOfUsedLayersPerCloud");
  maxFractionOfMissedLayersPerCloud = conf_.getParameter<double>("MaximalFractionOfMissedLayersPerCloud");
  maxFractionOfConsecutiveMissedLayersPerCloud = conf_.getParameter<double>("MaximalFractionOfConsecutiveMissedLayersPerCloud");
  increaseMaxNumberOfConsecutiveMissedLayersPerCloud = conf_.getParameter<unsigned int>("IncreaseMaxNumberOfConsecutiveMissedLayersPerCloud");
  increaseMaxNumberOfMissedLayersPerCloud = conf_.getParameter<unsigned int>("IncreaseMaxNumberOfMissedLayersPerCloud");
  
  doCleaning_ = conf.getParameter<bool>("DoCloudCleaning");
  mergingFraction_ = conf.getParameter<double>("MergingFraction");
  maxRecHitsInCloud_ = (unsigned int)conf.getParameter<int>("MaxRecHitsInCloud");
  scalefactorRoadSeedWindow_ = conf.getParameter<double>("scalefactorRoadSeedWindow");
  
  roadsLabel_ = conf.getParameter<std::string>("RoadsLabel");

}

RoadSearchCloudMakerAlgorithm::~RoadSearchCloudMakerAlgorithm() {

}

double RoadSearchCloudMakerAlgorithm::map_phi(double phi) {
  // map phi to [0,2pi]
  double result = phi;
  if ( result < -1.0*Geom::twoPi()) result = result + Geom::twoPi();
  if ( result < 0)                 result = Geom::twoPi() + result;
  if ( result > Geom::twoPi())     result = result - Geom::twoPi();
  return result;
}

double RoadSearchCloudMakerAlgorithm::map_phi2(double phi) {
  // map phi to [-pi,pi]
  double result = phi;
  if ( result < 1.0*Geom::pi() ) result = result + Geom::twoPi();
  if ( result >= Geom::pi())  result = result - Geom::twoPi();
  return result;
}

void RoadSearchCloudMakerAlgorithm::run(edm::Handle<RoadSearchSeedCollection> input,
                                        const SiStripRecHit2DCollection* rphiRecHits,
                                        const SiStripRecHit2DCollection* stereoRecHits,
                                        const SiStripMatchedRecHit2DCollection* matchedRecHits,
                                        const SiPixelRecHitCollection *pixRecHits,
                                        const edm::EventSetup& es,
                                        RoadSearchCloudCollection &output)
{  
  // intermediate arrays for storing clouds for cleaning
  const int nphibin = 24;
  const int netabin = 24;
  RoadSearchCloudCollection CloudArray[nphibin][netabin];
  
  // get roads
  edm::ESHandle<Roads> roads;
  es.get<RoadMapRecord>().get(roadsLabel_, roads);
  
  // get RoadSearchSeed collection
  const RoadSearchSeedCollection* inputSeeds = input.product();
  
  // set collections for general hit access method
  recHitVectorClass.setCollections(rphiRecHits,stereoRecHits,matchedRecHits,pixRecHits);
  recHitVectorClass.setMode(DetHitAccess::standard);
  
  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get hit matcher
  SiStripRecHitMatcher* theHitMatcher = new SiStripRecHitMatcher(3.0);

  edm::LogInfo("RoadSearch") << "Found " << inputSeeds->size() << " input seeds.";   
  // loop over seeds
  for ( RoadSearchSeedCollection::const_iterator seed = inputSeeds->begin(); seed != inputSeeds->end(); ++seed) {
    
    const Roads::RoadSeed *roadSeed = seed->getSeed();
      
    if ( roadSeed == 0 ) {
      edm::LogWarning("RoadSearch") << "RoadSeed could not be resolved from RoadSearchSeed hits, discard seed!";
    } else {
        
      Roads::type roadType = roads->getRoadType(roadSeed);
      if (NoFieldCosmic) roadType = Roads::RPhi;
        
      // fixme: from here on, calculate with 1st and 3rd seed hit (inner and outer of initial circle)
      // fixme: adapt to new seed structure
        
      // get global positions of the hits, calculate before Road lookup to be used
      const TrackingRecHit* innerSeedRingHit = (*(seed->begin()));
      const TrackingRecHit* outerSeedRingHit = (*(seed->end() - 1));
        
      GlobalPoint innerSeedHitGlobalPosition = tracker->idToDet(innerSeedRingHit->geographicalId())->surface().toGlobal(innerSeedRingHit->localPosition());
      GlobalPoint outerSeedHitGlobalPosition = tracker->idToDet(outerSeedRingHit->geographicalId())->surface().toGlobal(outerSeedRingHit->localPosition());
 
      LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " inner hit (x/y/z): "
			     << innerSeedHitGlobalPosition.x() << " / "
			     << innerSeedHitGlobalPosition.y() << " / "
			     << innerSeedHitGlobalPosition.z();
      LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " outer hit (x/y/z): "
			     << outerSeedHitGlobalPosition.x() << " / "
			     << outerSeedHitGlobalPosition.y() << " / "
			     << outerSeedHitGlobalPosition.z();
      
      LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " inner hit (r/phi): "
			     << innerSeedHitGlobalPosition.perp() << " / "
			     << innerSeedHitGlobalPosition.phi();
      LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " outer hit (r/phi): "
			     << outerSeedHitGlobalPosition.perp() << " / "
			     << outerSeedHitGlobalPosition.phi();


      // extrapolation parameters, phio: [0,2pi]
      double d0 = 0.0;
      double phi0 = -99.;
      double k0   = -99999999.99;
      
      double phi1 = -99.;
      double k1   = -99999999.99;
      // get bins of eta and phi of outer seed hit;
      
      double outer_phi = map_phi(outerSeedHitGlobalPosition.phi());
      double outer_eta = outerSeedHitGlobalPosition.eta();
      
      int phibin = (int)(nphibin*(outer_phi/(2*Geom::pi())));
      int etabin = (int)(netabin*(outer_eta+3.0)/6.0);
        
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
	    double x0=0.0; double y0=0.0;
	    double innerx=innerSeedHitGlobalPosition.x();
	    double innery=innerSeedHitGlobalPosition.y();
	    double outerx=outerSeedHitGlobalPosition.x();
	    double outery=outerSeedHitGlobalPosition.y();

	    if (NoFieldCosmic){
	      phi0=atan2(outery-innery,outerx-innerx);
	      double alpha=atan2(innery,innerx);
	      double d1=sqrt(innerx*innerx+innery*innery);
	      d0=d1*sin(alpha-phi0); x0=-d0*sin(phi0); y0=d0*cos(phi0); k0=0.0;
	    }else{
	      makecircle(innerx,innery,outerx,outery,x0,y0);
	      phi0 = phi0h;
	      k0 = omegah;
	    }              
	    LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " trajectory parameters: d0 = "<< d0 << " phi0 = " << phi0;
	  }
	}
      } else {
	double dz = outerSeedHitGlobalPosition.z() - innerSeedHitGlobalPosition.z();
	const double dz_min = 1.e-6; // cm;
	if ( std::abs(dz) < dz_min ) {
	  edm::LogWarning("RoadSearch") << "ZPhi road: seed Hits are less than .01 microns away in z, do not consider this seed.";
	} else {
	  // calculate z-phi extrapolation: phi = phi0 + k0 z
	  k0 = map_phi2(outerSeedHitGlobalPosition.phi() - innerSeedHitGlobalPosition.phi()) / dz;
	  phi0 =  map_phi(innerSeedHitGlobalPosition.phi() - k0 * innerSeedHitGlobalPosition.z());
	  
	  // get approx pt for use in correcting matched hits
	  makecircle(innerSeedHitGlobalPosition.x(),innerSeedHitGlobalPosition.y(),
		     outerSeedHitGlobalPosition.x(),outerSeedHitGlobalPosition.y(),
		     0.0,0.0); // x0,y0 = 0.0 for now
	  phi1 = phi0h;
	  k1 = omegah;
	}
      }
        
      // continue if valid extrapolation parameters have been found
      if ( (phi0 != -99.) && (k0 != -99999999.99) ) {
	const Roads::RoadSet *roadSet = seed->getSet();

	// create cloud
	RoadSearchCloud cloud;
          
	bool firstHitFound = false;
	unsigned int layerCounter = 0;
	unsigned int usedLayers = 0;
	unsigned int missedLayers = 0;
	unsigned int consecutiveMissedLayers = 0;

	unsigned int totalLayers = roadSet->size();

	// caluclate minNumberOfUsedLayersPerCloud, maxNumberOfMissedLayersPerCloud and maxNumberOfConsecutiveMissedLayersPerCloud 
	// by rounding to integer minFractionOfUsedLayersPerCloud. maxFractionOfMissedLayersPerCloud and maxFractionOfConsecutiveMissedLayersPerCloud
	unsigned int minNumberOfUsedLayersPerCloud = static_cast<unsigned int>(totalLayers * minFractionOfUsedLayersPerCloud + 0.5);
	if (minNumberOfUsedLayersPerCloud < 3) minNumberOfUsedLayersPerCloud = 3;
	unsigned int maxNumberOfMissedLayersPerCloud = static_cast<unsigned int>(totalLayers * maxFractionOfMissedLayersPerCloud + 0.5);
	unsigned int maxNumberOfConsecutiveMissedLayersPerCloud = static_cast<unsigned int>(totalLayers * maxFractionOfConsecutiveMissedLayersPerCloud + 0.5);

	// increase consecutive layer cuts between 0.9 and 1.5
	if (std::abs(outer_eta) > 0.9 && std::abs(outer_eta) < 1.5) {
	  maxNumberOfConsecutiveMissedLayersPerCloud += increaseMaxNumberOfConsecutiveMissedLayersPerCloud;
	  maxNumberOfMissedLayersPerCloud += increaseMaxNumberOfMissedLayersPerCloud;
	}
          
	for ( Roads::RoadSet::const_iterator roadSetVector = roadSet->begin();
	      roadSetVector != roadSet->end();
	      ++roadSetVector ) {
            
	  ++layerCounter;
	  unsigned int usedHitsInThisLayer = 0;
	  bool intersectsLayer = false;
            
	  for ( std::vector<const Ring*>::const_iterator ring = roadSetVector->begin(); ring != roadSetVector->end(); ++ring ) {
              
	    // calculate phi-range for lookup of DetId's in Rings of RoadSet
	    // calculate phi at radius of Ring using extrapolation, Ring radius average of Ring.rmin() and Ring.rmax()
	    double ringRadius = (*ring)->getrmin() + ((*ring)->getrmax()-(*ring)->getrmin())/2;
	    double ringZ      = (*ring)->getzmin() + ((*ring)->getzmax()-(*ring)->getzmin())/2;
	    double ringPhi = 0.0;
	    if ( roadType == Roads::RPhi ) {
	      ringPhi = phiFromExtrapolation(d0,phi0,k0,ringRadius,roadType);
	    } else {
	      ringPhi = phiFromExtrapolation(d0,phi0,k0,ringZ,roadType);
	    }
	    if (ringPhi == -99) continue;
	    intersectsLayer = true;

	    LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " testing ring at R = " << ringRadius
				   << " Z = " << ringZ << " ringPhi = " << ringPhi;
              
	    int nDetIds = (*ring)->getNumDetIds();
	    double theHalfRoad = theMinimumHalfRoad*(2.0*Geom::pi())/((double)nDetIds);
	    // calculate range in phi around ringPhi
	    double upperPhiRangeBorder = map_phi2(ringPhi + theHalfRoad);
	    double lowerPhiRangeBorder = map_phi2(ringPhi - theHalfRoad);

	    if ( lowerPhiRangeBorder <= upperPhiRangeBorder ) {
                
	      for ( Ring::const_iterator detid = (*ring)->lower_bound(lowerPhiRangeBorder); detid != (*ring)->upper_bound(upperPhiRangeBorder); ++detid) {
		usedHitsInThisLayer += FillRecHitsIntoCloudGeneral(detid->second,d0,phi0,k0,phi1,k1,roadType,ringPhi,
								   tracker.product(),theHitMatcher,cloud);
	      }
                
	    } else {
	      for ( Ring::const_iterator detid = (*ring)->lower_bound(lowerPhiRangeBorder); detid != (*ring)->end(); ++detid) {
		usedHitsInThisLayer += FillRecHitsIntoCloudGeneral(detid->second,d0,phi0,k0,phi1,k1,roadType,ringPhi,
								   tracker.product(),theHitMatcher,cloud);
	      }
                
	      for ( Ring::const_iterator detid = (*ring)->begin(); detid != (*ring)->upper_bound(upperPhiRangeBorder); ++detid) {
		usedHitsInThisLayer += FillRecHitsIntoCloudGeneral(detid->second,d0,phi0,k0,phi1,k1,roadType,ringPhi,
								   tracker.product(),theHitMatcher,cloud);
	      }
	    }
	  LogDebug("RoadSearch") << "Seed # " <<seed-inputSeeds->begin() << " now has " << usedHitsInThisLayer << "  hits in ring at R = " << ringRadius
				 << " Z = " << ringZ << " ringPhi = " << ringPhi;
	  }
            
	  if ( !firstHitFound ) {
	    if ( usedHitsInThisLayer > 0 ) {

	      firstHitFound = true;

	      // reset totalLayers according to first layer with hit
	      totalLayers = roadSet->size() - layerCounter + 1;

	      // re-caluclate minNumberOfUsedLayersPerCloud, maxNumberOfMissedLayersPerCloud and maxNumberOfConsecutiveMissedLayersPerCloud 
	      // by rounding to integer minFractionOfUsedLayersPerCloud. maxFractionOfMissedLayersPerCloud and maxFractionOfConsecutiveMissedLayersPerCloud
	      minNumberOfUsedLayersPerCloud = static_cast<unsigned int>(totalLayers * minFractionOfUsedLayersPerCloud + 0.5);
	      if (minNumberOfUsedLayersPerCloud < 3) minNumberOfUsedLayersPerCloud = 3;
	      maxNumberOfMissedLayersPerCloud = static_cast<unsigned int>(totalLayers * maxFractionOfMissedLayersPerCloud + 0.5);
	      maxNumberOfConsecutiveMissedLayersPerCloud = static_cast<unsigned int>(totalLayers * maxFractionOfConsecutiveMissedLayersPerCloud + 0.5);

	      // increase consecutive layer cuts between 0.9 and 1.5
	      if (std::abs(outer_eta) > 0.9 && std::abs(outer_eta) < 1.5) {
		maxNumberOfConsecutiveMissedLayersPerCloud += increaseMaxNumberOfConsecutiveMissedLayersPerCloud;
		maxNumberOfMissedLayersPerCloud += increaseMaxNumberOfMissedLayersPerCloud;
	      }

	      ++usedLayers;
	      consecutiveMissedLayers = 0;

	    }
	  } else {
	    if (intersectsLayer){
	      if ( usedHitsInThisLayer > 0 ) {
		++usedLayers;
		consecutiveMissedLayers = 0;
	      } else {
		++ missedLayers;
		++consecutiveMissedLayers;
	      }
	    }
	    LogDebug("RoadSearch") << "Seed # "<<seed-inputSeeds->begin() << " Layer info: " 
				   << " totalLayers: " << totalLayers 
				   << " usedLayers: " << usedLayers 
				   << " missedLayers: " << missedLayers
				   << " consecutiveMissedLayers: " << consecutiveMissedLayers;

	    // break condition, hole larger than maxNumberOfConsecutiveMissedLayersPerCloud
	    if ( consecutiveMissedLayers > maxNumberOfConsecutiveMissedLayersPerCloud ) {
 	      LogDebug("RoadSearch") << "BREAK: seed # "<<seed-inputSeeds->begin() 
				     << " More than " << maxNumberOfConsecutiveMissedLayersPerCloud << " missed consecutive layers!";
	      break;
	    }

	    // break condition, already  missed too many layers
	    if ( missedLayers > maxNumberOfMissedLayersPerCloud ) {
 	      LogDebug("RoadSearch") << "BREAK: seed # "<<seed-inputSeeds->begin() 
				     << " More than " << maxNumberOfMissedLayersPerCloud << " missed layers!";
	      break;
	    }

	    // break condition, cannot satisfy minimal number of used layers
	    if ( totalLayers-missedLayers < minNumberOfUsedLayersPerCloud ) {
 	      LogDebug("RoadSearch") << "BREAK: seed # "<<seed-inputSeeds->begin() 
				     << " Cannot satisfy at least " << minNumberOfUsedLayersPerCloud << " used layers!";
	      break;
	    }
	  }	  
            
	}

	if ( consecutiveMissedLayers <= maxNumberOfConsecutiveMissedLayersPerCloud ) {
	  if ( usedLayers >= minNumberOfUsedLayersPerCloud ) {
	    if ( missedLayers <= maxNumberOfMissedLayersPerCloud ) {

	      CloudArray[phibin][etabin].push_back(cloud);

	      if ( roadType == Roads::RPhi ){ 
		LogDebug("RoadSearch") << "This r-phi seed # "<<seed-inputSeeds->begin()
				       <<" yields a cloud with " <<cloud.size() <<" hits on " << usedLayers << " layers out of " << totalLayers;
	      } else {
		LogDebug("RoadSearch") << "This z-phi seed # "<<seed-inputSeeds->begin()
				       <<" yields a cloud with " <<cloud.size() <<" hits on " << usedLayers << " layers out of " << totalLayers;
	      }
	    } else {
 	      LogDebug("RoadSearch") << "Missed layers: " << missedLayers << " More than " << maxNumberOfMissedLayersPerCloud << " missed layers!";
	      if ( roadType == Roads::RPhi ){ 
		LogDebug("RoadSearch") << "This r-phi seed # "<<seed-inputSeeds->begin() <<" yields no clouds";
	      } else {
		LogDebug("RoadSearch") << "This z-phi seed # "<<seed-inputSeeds->begin() <<" yields no clouds";
	      }
	    }
	  }
 	  else {
 	    LogDebug("RoadSearch") << "Seed # "<<seed-inputSeeds->begin() <<" fails: used layers = " << usedLayers << " < " << minNumberOfUsedLayersPerCloud;
 	  }
	}
 	else {
 	  LogDebug("RoadSearch") << "Seed # "<<seed-inputSeeds->begin() <<" fails: consecutive missed layers = " << consecutiveMissedLayers << " > " << maxNumberOfConsecutiveMissedLayersPerCloud;
 	}
      }
    }
  }
  
  // Loop for initial cleaning
  for (int iphi=0; iphi<nphibin; ++iphi){
    for (int ieta=0; ieta<netabin; ++ieta){
      if (!CloudArray[iphi][ieta].empty()) {
        if (doCleaning_){
          RoadSearchCloudCollection temp = Clean(&CloudArray[iphi][ieta]);
          for ( RoadSearchCloudCollection::const_iterator ic = temp.begin(); ic!=temp.end(); ++ic)
            output.push_back(*ic);
        }
        else 
          for ( RoadSearchCloudCollection::const_iterator ic = CloudArray[iphi][ieta].begin(); 
                ic!=CloudArray[iphi][ieta].end(); ++ic)
            output.push_back(*ic);
      }
    }
  }

  delete theHitMatcher;
  edm::LogInfo("RoadSearch") << "Found " << output.size() << " clouds."; 
  for ( RoadSearchCloudCollection::const_iterator ic = output.begin(); ic!=output.end(); ++ic)
    edm::LogInfo("RoadSearch") << "    Cloud " << ic-output.begin()<< " has " << ic->size() << " hits."; 
  
}

unsigned int RoadSearchCloudMakerAlgorithm::FillRecHitsIntoCloudGeneral(DetId id, double d0, double phi0, double k0, double phi1, double k1, 
									Roads::type roadType, double ringPhi,
                                                                        const TrackerGeometry *tracker, const SiStripRecHitMatcher* theHitMatcher,
                                                                        RoadSearchCloud &cloud) {
  
  unsigned int usedRecHits = 0;
  
  bool double_ring_layer = !isSingleLayer(id);
  
  std::vector<TrackingRecHit*> recHitVector = recHitVectorClass.getHitVector(&id);
  
  for ( std::vector<TrackingRecHit*>::const_iterator recHitIterator = recHitVector.begin(); recHitIterator != recHitVector.end(); ++recHitIterator) {
    
    if (      (unsigned int)id.subdetId() == StripSubdetector::TIB 
              || (unsigned int)id.subdetId() == StripSubdetector::TOB 
              || (unsigned int)id.subdetId() == StripSubdetector::TID 
              || (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
      
      const SiStripRecHit2D *recHit = (SiStripRecHit2D*)(*recHitIterator);
      DetId hitId = recHit->geographicalId();

      LogDebug("RoadSearch") << "    Testing hit at (x/y/z): " 
			     << tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition()).x() << " / " 
			     << tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition()).y() << " / " 
			     << tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition()).z();

      if ( roadType == Roads::RPhi ) {
        if (double_ring_layer && isSingleLayer(hitId)) {
          //
          //  This is where the barrel stereoRecHits end up for Roads::RPhi
          //
          
          // Adjust matched hit for track angle

          const GluedGeomDet *theGluedDet = dynamic_cast<const GluedGeomDet*>(tracker->idToDet(hitId));

          SiStripMatchedRecHit2D* theCorrectedHit = CorrectMatchedHit(*recHitIterator,theGluedDet,
                                                                        tracker, theHitMatcher,
                                                                        k0, phi0);
          if (theCorrectedHit != 0){

            GlobalPoint ghit = tracker->idToDet(theCorrectedHit->geographicalId())->surface().toGlobal(theCorrectedHit->localPosition());
            double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
            double hitphi = map_phi(ghit.phi());
            double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
            
            float dp = (hitphi-phi);
            float dx = hitRadius*tan(dp);
            
	    LogDebug("RoadSearch") << "   Hit phi = " << hitphi << " expected phi = " << phi
				       <<"  dx = " << dx << " for dxMax = " << phiMax(roadType,phi0,k0);

            // switch cut to dx instead of dphi
            if ( std::abs(dx) < phiMax(roadType,phi0,k0) ) {
              if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
                cloud.addHit(recHit);
                ++usedRecHits;
              }
            }
            delete theCorrectedHit;
	  }
	} else { // Single layer hits here
	  if ( isBarrelSensor(hitId) ) {
	    //
	    //  This is where the barrel rphiRecHits end up for Roads::RPhi
	    //
	    GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
	    double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
	    double hitphi = map_phi(ghit.phi());
	    double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
              
	    float dp = (hitphi-phi);
	    float dx = hitRadius*tan(dp);
	    LogDebug("RoadSearch") << "   Hit phi = " << hitphi << " expected phi = " << phi
				       <<"  dx = " << dx << " for dxMax = " << phiMax(roadType,phi0,k0);
	    // switch cut to dx instead of dphi
	    if ( std::abs(dx) < phiMax(roadType,phi0,k0) ) {
              if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		cloud.addHit(recHit);
		++usedRecHits;
	      }
	    }
	  } 
	  else {

	    LocalPoint hit = recHit->localPosition();
	    const StripTopology *topology = dynamic_cast<const StripTopology*>(&(tracker->idToDetUnit(hitId)->topology()));
	    double stripAngle = topology->stripAngle(topology->strip(hit));
	    double stripLength = topology->localStripLength(hit);

	    LocalPoint innerHitLocal(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	    LocalPoint outerHitLocal(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);

	    double innerRadius = tracker->idToDetUnit(hitId)->surface().toGlobal(innerHitLocal).perp(); 
	    double outerRadius = tracker->idToDetUnit(hitId)->surface().toGlobal(outerHitLocal).perp();
	    double innerExtrapolatedPhi = phiFromExtrapolation(d0,phi0,k0,innerRadius,roadType);
	    double outerExtrapolatedPhi = phiFromExtrapolation(d0,phi0,k0,outerRadius,roadType);

	    GlobalPoint innerHitGlobal =tracker->idToDetUnit(hitId)->surface().toGlobal(innerHitLocal); 
	    GlobalPoint outerHitGlobal =tracker->idToDetUnit(hitId)->surface().toGlobal(outerHitLocal); 

	    GlobalPoint innerRoadGlobal(GlobalPoint::Cylindrical(innerRadius,innerExtrapolatedPhi,
								 tracker->idToDetUnit(hitId)->surface().toGlobal(hit).z()));
	    GlobalPoint outerRoadGlobal(GlobalPoint::Cylindrical(outerRadius,outerExtrapolatedPhi,
								 tracker->idToDetUnit(hitId)->surface().toGlobal(hit).z()));

	    LocalPoint innerRoadLocal = tracker->idToDetUnit(hitId)->surface().toLocal(innerRoadGlobal);
	    LocalPoint outerRoadLocal = tracker->idToDetUnit(hitId)->surface().toLocal(outerRoadGlobal);

	    double dxinter = CheckXYIntersection(innerHitLocal, outerHitLocal, 
					       innerRoadLocal, outerRoadLocal);

	    LogDebug("RoadSearch") << " Hit phi inner = " << innerHitGlobal.phi() << " and outer = " << outerHitGlobal.phi()
				   << " expected inner phi = " << innerExtrapolatedPhi
				   << " and outer phi = "      << outerExtrapolatedPhi
				   <<"  dx = " << dxinter << " for dxMax = " << phiMax(roadType,phi0,k0);

	    if ( fabs(dxinter) < phiMax(roadType,phi0,k0)) {
	      //
	      //  This is where the disk rphiRecHits end up for Roads::ZPhi
	      //
	      if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		cloud.addHit(recHit);
		++usedRecHits;
	      }
	    }
	    //else
	      //std::cout<< " ===>>> HIT FAILS !!! " << std::endl;
	  }
	} 
      } else {
	//
	// roadType == Roads::ZPhi
	//
	if (double_ring_layer && isSingleLayer(hitId)) {

          // Adjust matched hit for track angle

          //const SiStripMatchedRecHit2D *theRH = dynamic_cast<SiStripMatchedRecHit2D*>(*recHitIterator);
          const GluedGeomDet *theGluedDet = dynamic_cast<const GluedGeomDet*>(tracker->idToDet(hitId));

          SiStripMatchedRecHit2D* theCorrectedHit = CorrectMatchedHit(*recHitIterator,theGluedDet,
                                                                        tracker, theHitMatcher,
                                                                        k1, phi1);
          if (theCorrectedHit != 0){

            GlobalPoint ghit = tracker->idToDet(theCorrectedHit->geographicalId())->surface().toGlobal(theCorrectedHit->localPosition());
            double hitphi = map_phi(ghit.phi());
            double hitZ = ghit.z();
	    double phi = phiFromExtrapolation(d0,phi0,k0,hitZ,roadType);
            
	    float dp = hitphi-phi;
	    float dx = hitZ*tan(dp);
              
	    //
	    //  This is where the disk stereoRecHits end up for Roads::ZPhi
	    //
	    if ( std::abs(dx) < phiMax(roadType,phi0,k1)) {
	      if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		cloud.addHit(recHit);
		++usedRecHits;
	      }
	    }
	    delete theCorrectedHit;
	  }
	} else { // Single layer hits here
	  if ( isBarrelSensor(hitId) ) {
	    //
	    //  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
	    //
	    LocalPoint hit = recHit->localPosition();
	    const StripTopology *topology = dynamic_cast<const StripTopology*>(&(tracker->idToDetUnit(hitId)->topology()));
	    double stripAngle = topology->stripAngle(topology->strip(hit));
	    double stripLength = topology->localStripLength(hit);

	    //if (stripAngle!=0) std::cout<<"HEY, WE FOUND A HIT ON A STEREO MODULE!!!" << std::endl;
	    // new method
	    LocalPoint innerHitLocal(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	    LocalPoint outerHitLocal(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	    double innerHitPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(innerHitLocal).phi()); 
	    double outerHitPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(outerHitLocal).phi());
	    double innerHitZ = tracker->idToDetUnit(hitId)->surface().toGlobal(innerHitLocal).z(); 
	    double outerHitZ = tracker->idToDetUnit(hitId)->surface().toGlobal(outerHitLocal).z();
	    double innerExtrapolatedPhi = phiFromExtrapolation(d0,phi0,k0,innerHitZ,roadType);
	    double outerExtrapolatedPhi = phiFromExtrapolation(d0,phi0,k0,outerHitZ,roadType);

	    double midPointZ = 0.5*(innerHitZ+outerHitZ);

	    double dPhiInter = CheckZPhiIntersection(innerHitPhi, innerHitZ, outerHitPhi, outerHitZ, 
						   innerExtrapolatedPhi, innerHitZ,
						   outerExtrapolatedPhi, outerHitZ);

	    double dX = midPointZ*tan(dPhiInter);

	    if (std::abs(dX) < 1.5*phiMax(roadType,phi0,k1)) {
	      if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		cloud.addHit(recHit);
		++usedRecHits;
	      }
	    }

	  } else {

	    //
	    //  This is where the disk rphiRecHits end up for Roads::ZPhi
	    //
	    LocalPoint hit = recHit->localPosition();
	    const StripTopology *topology = dynamic_cast<const StripTopology*>(&(tracker->idToDetUnit(hitId)->topology()));
	    double stripAngle = topology->stripAngle(topology->strip(hit));
	    double stripLength = topology->localStripLength(hit);
	    // new method
	    double hitZ = tracker->idToDetUnit(hitId)->surface().toGlobal(hit).z(); 
	    double extrapolatedPhi = phiFromExtrapolation(d0,phi0,k0,hitZ,roadType);

	    LocalPoint innerHitLocal(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	    LocalPoint outerHitLocal(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);

	    double innerHitPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(innerHitLocal).phi()); 
	    double outerHitPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(outerHitLocal).phi());
	    //double innerZ = tracker->idToDetUnit(hitId)->surface().toGlobal(innerHitLocal).z(); 
	    //double outerZ = tracker->idToDetUnit(hitId)->surface().toGlobal(outerHitLocal).z();
	    //if (innerZ != outerZ)  std::cout<<"HEY!!! innerZ = " << innerZ << " != outerZ = " << outerZ << std::endl;

	    double deltaPhi = ZPhiDeltaPhi(innerHitPhi,outerHitPhi,extrapolatedPhi);
	    double deltaX   =  hitZ*tan(deltaPhi);
	    if (std::abs(deltaX) < phiMax(roadType,phi0,k1)){
	      if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		cloud.addHit(recHit);
		++usedRecHits;
	      }
	    }
	  }
	} 
      }
    } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel 
                || (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap) {
      if ( UsePixels ) {
        
        const SiPixelRecHit *recHit = (SiPixelRecHit*)(*recHitIterator);
        
        if ( roadType == Roads::RPhi ) {
          
          if ( isBarrelSensor(id) ) {
            // Barrel Pixel, RoadType RPHI
            
            GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
            double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
            double hitphi = map_phi(ghit.phi());
            double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
            
            float dp = hitphi-phi;
            float dx = hitRadius*tan(dp);
            
            // switch cut to dx instead of dphi
            if ( std::abs(dx) < phiMax(roadType,phi0,k0) ) {
              cloud.addHit(recHit);
              ++usedRecHits;
            }
	  } else {
              
	    // Forward Pixel,roadtype RPHI
              
	    // Get Local Hit Position of the Pixel Hit
	    LocalPoint hit = recHit->localPosition();
              
	    // Get Phi of hit position 
	    double hitphi = map_phi(tracker->idToDetUnit(id)->surface().toGlobal(hit).phi());
              
	    // Get Global Hit position
	    GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
              
	    // Get Hit Radis
	    double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
              
	    // Get Phi from extrapolation
	    double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
              
	    if ( std::abs(hitphi-phi) < phiMax(roadType,phi0,k0) ) {
	      cloud.addHit(recHit);
	      ++usedRecHits;
	    }	
	  }
	} else {
            
	  GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());            
	  double  phi = phiFromExtrapolation(d0,phi0,k0,ghit.z(),roadType);            
	  double hitphi = map_phi(ghit.phi());
	  double dphi = map_phi2(hitphi-phi);
	  float dx = ghit.z()*tan(dphi);
            
	  if ( std::abs(dx) < 0.25 ) {
	    cloud.addHit(recHit);
	    ++usedRecHits;
	  }
	}
      }
    } else {
      edm::LogError("RoadSearch") << "recHitVector from general hit access function contains unknown detector id: " << (unsigned int)id.subdetId() << " rawId: " << id.rawId();
    }
    
  } //for loop over all recHits
  
  
  return usedRecHits;
}

unsigned int RoadSearchCloudMakerAlgorithm::FillPixRecHitsIntoCloud(DetId id, const SiPixelRecHitCollection *inputRecHits, 
                                                                    double d0, double phi0, double k0, Roads::type roadType, double ringPhi,
                                                                    const TrackerGeometry *tracker, RoadSearchCloud &cloud) {
  
  
  unsigned int usedRecHits = 0;
  
  // Get Geometry
  //const PixelTopology *topology = dynamic_cast<const PixelTopology*>(&(tracker->idToDetUnit(id)->topology()));
  
  
  // retrieve vector<SiPixelRecHit> for id
  // loop over SiPixelRecHit
  // check if compatible with cloud, fill into cloud
  
  SiPixelRecHitCollection::const_iterator recHitMatch = inputRecHits->find(id);
  if (recHitMatch == inputRecHits->end()) return usedRecHits;

  const SiPixelRecHitCollection::DetSet recHitRange = *recHitMatch;
  
  for ( SiPixelRecHitCollection::DetSet::const_iterator recHitIterator = recHitRange.begin(); 
        recHitIterator != recHitRange.end(); ++recHitIterator) {
    
    const SiPixelRecHit * recHit = &(*recHitIterator);
    
    if ( roadType == Roads::RPhi ) {
      
      if ( isBarrelSensor(id) ) {
        // Barrel Pixel, RoadType RPHI
        
        GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
        double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
        double hitphi = map_phi(ghit.phi());
        double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
        
        if ( std::abs(hitphi-phi) < phiMax(roadType,phi0,k0) ) {
          cloud.addHit(recHit);
          ++usedRecHits;
        }
      } 
      else {
        
        // Forward Pixel,roadtype RPHI
        
        // Get Local Hit Position of the Pixel Hit
        LocalPoint hit = recHit->localPosition();
        
        // Get Phi of hit position 
        double hitphi = map_phi(tracker->idToDetUnit(id)->surface().toGlobal(hit).phi());
        
        // Get Global Hit position
        GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
        
        // Get Hit Radis
        double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
        
        // Get Phi from extrapolation
        double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
        
        if ( std::abs(hitphi-phi) < phiMax(roadType,phi0,k0) ) {
          cloud.addHit(recHit);
          ++usedRecHits;
        }	
      }
    } 
    
    else {
      
      GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
      
      double  phi = phiFromExtrapolation(d0,phi0,k0,ghit.z(),roadType);
      if ( (phi - phiMax(roadType,phi0,k0)) < ringPhi && (phi + phiMax(roadType,phi0,k0))>ringPhi ) {
        cloud.addHit(recHit);
        ++usedRecHits;
      }
    }
    
  }
  
  return usedRecHits;
}

bool RoadSearchCloudMakerAlgorithm::isSingleLayer(DetId id) {
  
  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    if ( !tibid.glued() ) {
      return true;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    if ( !tobid.glued() ) {
      return true;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    if ( !tidid.glued() ) {
      return true;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    if ( !tecid.glued() ) {
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

double RoadSearchCloudMakerAlgorithm::phiFromExtrapolation(double d0, double phi0, double k0, double ringRadius, Roads::type roadType) {
  
  double ringPhi = -99.;
  if ( roadType == Roads::RPhi ) {
    double omega=k0, rl=ringRadius;
    double sp0=sin(phi0); double cp0=cos(phi0);  
    if (fabs(omega)>0.000005){
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
    }
    else {
      double cee = rl*rl - d0*d0 -0.25*omega*omega - omega*d0;
      if (cee<0.0){return ringPhi;}
      double l = sqrt(cee);
      double xh=-sp0*d0+l*cp0-0.5*l*l*omega*sp0;
      double yh= cp0*d0+l*sp0+0.5*l*l*omega*cp0;
      double phih=atan2(yh,xh);
      ringPhi = map_phi(phih);     
    }
  } 
  else {
    ringPhi = map_phi(phi0 + k0 * ringRadius);
  }
  
  return ringPhi;
}

double RoadSearchCloudMakerAlgorithm::phiMax(Roads::type roadType,
                                             double phi0, double k0) {
  
  double dphi;
  if ( roadType == Roads::RPhi ) {
    // switch cut to dx instead of dphi
    // Still call this dphi, but will now be dx
    dphi = theRPhiRoadSize + 0.15*82.0*fabs(k0);
  }
  else if ( roadType == Roads::ZPhi ) {
    dphi = theZPhiRoadSize + 0.4*82.0*fabs(k0);
  }
  else {
    edm::LogWarning("RoadSearch") << "Bad roadType: "<< roadType;
    dphi = theZPhiRoadSize;
  }
  return dphi;
  
}

void RoadSearchCloudMakerAlgorithm::makecircle(double x1, double y1, 
                                               double x2,double y2, double x3, double y3){
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
  phi0h=atan2(yc,xc)+s3*Geom::halfPi();
  omegah=-s3/rc;
}

double RoadSearchCloudMakerAlgorithm::CheckXYIntersection(LocalPoint& inner1, LocalPoint& outer1,
							LocalPoint& inner2, LocalPoint& outer2){

  double deltaX = -999.;
  // just get the x coord of intersection of two line segments
  // check if intersection lies inside segments
  double det12 = inner1.x()*outer1.y() - inner1.y()*outer1.x();
  double det34 = inner2.x()*outer2.y() - inner2.y()*outer2.x();

  double xinter = (det12*(inner2.x()-outer2.x()) - det34*(inner1.x()-outer1.x()))/
    ((inner1.x()-outer1.x())*(inner2.y()-outer2.y()) - 
     (inner2.x()-outer2.x())*(inner1.y()-outer1.y()));

  bool inter = true;
  if (inner1.x() < outer1.x()){
    if ((xinter<inner1.x()) || (xinter>outer1.x())) inter = false;
  }
  else{
    if ((xinter>inner1.x()) || (xinter<outer1.x())) inter = false;
  }

  if (inner2.x() < outer2.x()){
    if ((xinter<inner2.x()) || (xinter>outer2.x())) inter = false;
  }
  else{
    if ((xinter>inner2.x()) || (xinter<outer2.x())) inter = false;
  }

  if (inter){
    deltaX = 0;
  }
  else{
    deltaX = min(fabs(inner1.x()-inner2.x()),fabs(outer1.x()-outer2.x()));
  }
  return deltaX;

}

double RoadSearchCloudMakerAlgorithm::CheckZPhiIntersection(double iPhi1, double iZ1, double oPhi1, double oZ1,
							    double iPhi2, double iZ2, double oPhi2, double oZ2){
  
  // Have to make sure all are in the same hemisphere
  if ((iPhi1 > Geom::pi() || oPhi1 > Geom::pi() || iPhi2 > Geom::pi() || oPhi2 > Geom::pi()) &&
      (iPhi1 < Geom::pi() || oPhi1 < Geom::pi() || iPhi2 < Geom::pi() || oPhi2 < Geom::pi())){
    iPhi1 = map_phi2(iPhi1);  oPhi1 = map_phi2(oPhi1);
    iPhi2 = map_phi2(iPhi2);  oPhi2 = map_phi2(oPhi2);
  }

  double deltaPhi = -999.;
  // just get the x coord of intersection of two line segments
  // check if intersection lies inside segments
  double det12 = iZ1*oPhi1 - iPhi1*oZ1;
  double det34 = iZ2*oPhi2 - iPhi2*oZ2;

  double xinter = (det12*(iZ2-oZ2) - det34*(iZ1-oZ1))/
    ((iZ1-oZ1)*(iPhi2-oPhi2) - 
     (iZ2-oZ2)*(iPhi1-oPhi1));

  bool inter = true;
  if (iZ1 < oZ1){
    if ((xinter<iZ1) || (xinter>oZ1)) inter = false;
  }
  else{
    if ((xinter>iZ1) || (xinter<oZ1)) inter = false;
  }

  if (iZ2 < oZ2){
    if ((xinter<iZ2) || (xinter>oZ2)) inter = false;
  }
  else{
    if ((xinter>iZ2) || (xinter<oZ2)) inter = false;
  }

  if (inter){
    deltaPhi = 0;
  }
  else{
    deltaPhi = min(fabs(iPhi2-iPhi1),fabs(oPhi2-oPhi1));
  }
  return deltaPhi;

}


double RoadSearchCloudMakerAlgorithm::ZPhiDeltaPhi(double hitPhi1, double hitPhi2, double predictedPhi){

  double deltaPhi = -999.;

  double dPhiHits = map_phi2(hitPhi1-hitPhi2);
  double dPhi1 = map_phi2(hitPhi1-predictedPhi);
  double dPhi2 = map_phi2(hitPhi2-predictedPhi);

  if (dPhiHits >= 0){  // hitPhi1 >= hitPhi2
    if ( (dPhi1>=0.0) && (dPhi2 <= 0.0))
      deltaPhi = 0.0;
    else{
      if (std::abs(dPhi1)<std::abs(dPhi2))
	deltaPhi = dPhi1;
      else
	deltaPhi = dPhi2;
    }
  }
  else { // hitPhi1 < hitPhi2
    if ( (dPhi1<=0.0) && (dPhi2 >= 0.0))
      deltaPhi = 0.0;
    else{
      if (std::abs(dPhi1)<std::abs(dPhi2))
	deltaPhi = dPhi1;
      else
	deltaPhi = dPhi2;
    }
  }

  return deltaPhi;

}

RoadSearchCloudCollection RoadSearchCloudMakerAlgorithm::Clean(RoadSearchCloudCollection* inputCollection){
  
  RoadSearchCloudCollection output;
  
  //
  //  no raw clouds - nothing to try merging
  //

  if ( inputCollection->empty() ){
    LogDebug("RoadSearch") << "Found " << output.size() << " clean clouds.";
    return output;  
  }
  
  //
  //  1 raw cloud - nothing to try merging, but one cloud to duplicate
  //
  
  if ( 1==inputCollection->size() ){
    output = *inputCollection;
//     RoadSearchCloud *temp = inputCollection->begin()->clone();
//     output.push_back(*temp);
//     delete temp;
    LogDebug("RoadSearch") << "Found " << output.size() << " clean clouds.";
    return output;
  }  
  
  //
  //  got > 1 raw cloud - something to try merging
  //
  std::vector<bool> already_gone(inputCollection->size());
  for (unsigned int i=0; i<inputCollection->size(); ++i) {
    already_gone[i] = false; 
  }
  
  int raw_cloud_ctr=0;
  // loop over clouds
  for ( RoadSearchCloudCollection::const_iterator raw_cloud = inputCollection->begin(); raw_cloud != inputCollection->end(); ++raw_cloud) {
    ++raw_cloud_ctr;
    
    if (already_gone[raw_cloud_ctr-1])continue;
    
    // produce output cloud where other clouds are merged in
    // create temp pointer for clone which will be deleted afterwards
//     RoadSearchCloud *temp_lone_cloud = raw_cloud->clone();
//     RoadSearchCloud lone_cloud = *temp_lone_cloud;
    RoadSearchCloud lone_cloud = *raw_cloud;

    int second_cloud_ctr=raw_cloud_ctr;
    for ( RoadSearchCloudCollection::const_iterator second_cloud = raw_cloud+1; second_cloud != inputCollection->end(); ++second_cloud) {
      second_cloud_ctr++;
      
      std::vector<const TrackingRecHit*> unshared_hits;
      
      if ( already_gone[second_cloud_ctr-1] )continue;
      
      for ( RoadSearchCloud::RecHitVector::const_iterator second_cloud_hit = second_cloud->begin_hits();
            second_cloud_hit != second_cloud->end_hits();
            ++ second_cloud_hit ) {
        bool is_shared = false;
        for ( RoadSearchCloud::RecHitVector::const_iterator lone_cloud_hit = lone_cloud.begin_hits();
              lone_cloud_hit != lone_cloud.end_hits();
              ++ lone_cloud_hit ) {
          
          if ((*lone_cloud_hit)->geographicalId() == (*second_cloud_hit)->geographicalId())
            if ((*lone_cloud_hit)->localPosition().x() == (*second_cloud_hit)->localPosition().x())
              if ((*lone_cloud_hit)->localPosition().y() == (*second_cloud_hit)->localPosition().y())
		{is_shared=true; break;}
        }
	if (!is_shared)  unshared_hits.push_back(*second_cloud_hit);
          
	if ( ((float(unshared_hits.size())/float(lone_cloud.size())) > 
	      ((float(second_cloud->size())/float(lone_cloud.size()))-mergingFraction_)) &&
	     ((float(unshared_hits.size())/float(second_cloud->size())) > (1-mergingFraction_))){
	  // You'll never merge these clouds..... Could quit now!
	  break;
	}
          
	if (lone_cloud.size()+unshared_hits.size() > maxRecHitsInCloud_) {
	  break;
	}
          
      }
        
      double f_lone_shared=double(second_cloud->size()-unshared_hits.size())/double(lone_cloud.size());
      double f_second_shared=double(second_cloud->size()-unshared_hits.size())/double(second_cloud->size());
      
      if ( ( (static_cast<unsigned int>(f_lone_shared*1E9) > static_cast<unsigned int>(mergingFraction_*1E9))||(static_cast<unsigned int>(f_second_shared*1E9) > static_cast<unsigned int>(mergingFraction_*1E9)) ) 
	   && (lone_cloud.size()+unshared_hits.size() <= maxRecHitsInCloud_) ){
	
	LogDebug("RoadSearch") << " Merge CloudA: " << raw_cloud_ctr << " with  CloudB: " << second_cloud_ctr 
			       << " Shared fractions are " << f_lone_shared << " and " << f_second_shared;
          
	//
	//  got a cloud to merge
	//
	for (unsigned int k=0; k<unshared_hits.size(); ++k) {
	  lone_cloud.addHit(unshared_hits[k]);
	}
          
	already_gone[second_cloud_ctr-1]=true;
        
      }//end got a cloud to merge
      
    }//interate over all second clouds
      
    output.push_back(lone_cloud);
    
  }//iterate over all raw clouds
  
  LogDebug("RoadSearch") << "Found " << output.size() << " clean clouds.";
  
  return output;
}

SiStripMatchedRecHit2D* RoadSearchCloudMakerAlgorithm::CorrectMatchedHit(const TrackingRecHit *originalHit,
                                                                         const GluedGeomDet* gluedDet,
                                                                         const TrackerGeometry *tracker,
                                                                         const SiStripRecHitMatcher* theHitMatcher,
                                                                         double k0, double phi0) {
          // VI January 2012
          // this is not supported anymore w/o cpe
    
          const SiStripMatchedRecHit2D *theRH = dynamic_cast<const SiStripMatchedRecHit2D*>(originalHit);
          if (theRH == 0) {
            std::cout<<" Could not cast original hit" << std::endl;
          }
          if (theRH != 0){
            const GeomDet *recHitGeomDet = tracker->idToDet(theRH->geographicalId());
            const GluedGeomDet *theGluedDet = dynamic_cast<const GluedGeomDet*>(recHitGeomDet);
             
            const GeomDetUnit* theMonoDet = theGluedDet->monoDet();
            const SiStripRecHit2D theMonoHit   = theRH->monoHit();
            //GlobalPoint monoRHpos = (theMonoDet->surface()).toGlobal(theMonoHit.localPosition());
             
            GlobalPoint gcenterofstrip=(theMonoDet->surface()).toGlobal(theMonoHit.localPosition());
             
            float gtrackangle_xy = map_phi2(phi0 + 2.0*asin(0.5*gcenterofstrip.perp()*k0));
            float rzangle = atan2(gcenterofstrip.perp(),gcenterofstrip.z());
 
            GlobalVector gtrackangle2(cos(gtrackangle_xy)*sin(rzangle),
                                      sin(gtrackangle_xy)*sin(rzangle),
                                      cos(rzangle));
            LocalVector trackdirection2=((tracker->idToDet(theRH->geographicalId()))->surface()).toLocal(gtrackangle2);
            //GlobalVector gdir = theMonoDet->surface().toGlobal(trackdirection2);
 
            SiStripMatchedRecHit2D* theCorrectedHit = theHitMatcher->match(theRH,theGluedDet,trackdirection2);
            if (theCorrectedHit!=0) return theCorrectedHit;
          }
 
          return 0;
}

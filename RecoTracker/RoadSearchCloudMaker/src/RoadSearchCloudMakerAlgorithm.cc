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
// $Author: burkett $
// $Date: 2007/03/29 22:14:01 $
// $Revision: 1.39 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/RoadSearchCloudMaker/interface/RoadSearchCloudMakerAlgorithm.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
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
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchDetIdHelper.h"

#include "RecoTracker/RoadMapRecord/interface/RoadMapRecord.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

using namespace std;

double RoadSearchCloudMakerAlgorithm::epsilon      =   0.000000001;
//double RoadSearchCloudMakerAlgorithm::half_pi      =   1.570796327;

RoadSearchCloudMakerAlgorithm::RoadSearchCloudMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
  recHitVectorClass.setMode(DetHitAccess::standard);    
  recHitVectorClass.use_rphiRecHits(conf_.getParameter<bool>("UseRphiRecHits"));
  recHitVectorClass.use_stereoRecHits(conf_.getParameter<bool>("UseStereoRecHits"));
  
  
  theRPhiRoadSize =  conf_.getParameter<double>("RPhiRoadSize");
  theZPhiRoadSize =  conf_.getParameter<double>("ZPhiRoadSize");
  UsePixels = conf_.getParameter<bool>("UsePixelsinRS");
  NoFieldCosmic = conf_.getParameter<bool>("StraightLineNoBeamSpotCloud");
  theMinimumHalfRoad = conf_.getParameter<double>("MinimumHalfRoad");
  
  maxDetHitsInCloudPerDetId = (unsigned int)conf_.getParameter<int>("MaxDetHitsInCloudPerDetId");
  minNumberOfUsedLayersPerRoad = (unsigned int)conf_.getParameter<int>("MinimalNumberOfUsedLayersPerRoad");
  maxNumberOfMissedLayersPerRoad = (unsigned int)conf_.getParameter<int>("MaximalNumberOfMissedLayersPerRoad");
  
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

void RoadSearchCloudMakerAlgorithm::run(edm::Handle<TrajectorySeedCollection> input,
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
  
  // get trajectoryseed collection
  const TrajectorySeedCollection* inputSeeds = input.product();
  
  // load the DetIds of the hits
  const std::vector<DetId> availableIDs = rphiRecHits->ids();
  const std::vector<DetId> availableIDs2 = stereoRecHits->ids();
  const std::vector<DetId> availableIDs3 = pixRecHits->ids();
  
  // set collections for general hit access method
  recHitVectorClass.setCollections(rphiRecHits,stereoRecHits,matchedRecHits,pixRecHits);
  recHitVectorClass.setMode(DetHitAccess::standard);
  
  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get hit matcher
  SiStripRecHitMatcher* theHitMatcher = new SiStripRecHitMatcher(3.0);
  
  // counter for seeds for edm::Ref size_type
  int seedCounter = -1;
  
  // loop over seeds
  for ( TrajectorySeedCollection::const_iterator seed = inputSeeds->begin(); seed != inputSeeds->end(); ++seed) {
    
    ++seedCounter;
    
    // get DetIds of SiStripRecHit2D of Seed, assuming first is hit from inner SeedRing, second is hit from outer SeedRing
    if ( seed->nHits() < 2 ) {
      edm::LogError("RoadWarning") << "Seed has less then two linked TrackingRecHit, do not consider this seed.";
    } else {
      
      std::vector<DetId> seedRingDetIds;
      std::vector<double> seedRingHitsPhi;
      
      for ( TrajectorySeed::const_iterator hit = seed->recHits().first;
            hit != seed->recHits().second;
            ++hit ) {
        seedRingDetIds.push_back(hit->geographicalId());
        seedRingHitsPhi.push_back(map_phi2(tracker->idToDet(hit->geographicalId())->surface().toGlobal(hit->localPosition()).phi()));
      }
      
      //       output_ << "Input detids: ";
      //       for ( std::vector<DetId>::const_iterator id = seedRingDetIds.begin();
      // 	    id != seedRingDetIds.end();
      // 	    ++id ) {
      // 	output_ << roads->getRing(RoadSearchDetIdHelper::ReturnRPhiId(*id))->getindex() << " ";
      //       }
      //       output_ << "\n";
      
      // get RoadSeed from Roads
      // GlobalPoint returns phi in [-pi,pi] but rings are mapped in [0,2pi]
      const Roads::RoadSeed *roadSeed = roads->getRoadSeed(seedRingDetIds, seedRingHitsPhi, scalefactorRoadSeedWindow_);
      
      if ( roadSeed == 0 ) {
        edm::LogWarning("RoadSearch") << "RoadSeed could not be resolved from TrajectorySeed hits, discard seed!";
      } else {
        
        const Roads::type roadType = roads->getRoadType(roadSeed);
        
        //       output_ << "Inner Seed Rings: ";
        //       // print inner and outer seed ring indices
        //       for ( std::vector<const Ring*>::const_iterator innerSeedRing = roadSeed->first.begin();
        // 	    innerSeedRing != roadSeed->first.end();
        // 	    ++innerSeedRing ) {
        // 	output_ << (*innerSeedRing)->getindex() << " ";
        //       }
        //       output_ << "\n";
        
        //       output_ << "Outer Seed Rings: ";
        //       // print outer and outer seed ring indices
        //       for ( std::vector<const Ring*>::const_iterator outerSeedRing = roadSeed->second.begin();
        // 	    outerSeedRing != roadSeed->second.end();
        // 	    ++outerSeedRing ) {
        // 	output_ << (*outerSeedRing)->getindex() << " ";
        //       }
        //       output_ << "\n";
        
        
        // fixme: from here on, calculate with 1st and 3rd seed hit (inner and outer of initial circle)
        // fixme: adapt to new seed structure
        
        // get global positions of the hits, calculate before Road lookup to be used
        const TrackingRecHit* innerSeedRingHit = 0;
        const TrackingRecHit* outerSeedRingHit = 0;
        if ( seed->nHits() >= 3 ) {
          innerSeedRingHit = &(*(seed->recHits().first));
          outerSeedRingHit = &(*(seed->recHits().first + 2));
        } else {
          innerSeedRingHit = &(*(seed->recHits().first));
          outerSeedRingHit = &(*(seed->recHits().second-1));
        }
        
        GlobalPoint innerSeedHitGlobalPosition = tracker->idToDet(innerSeedRingHit->geographicalId())->surface().toGlobal(innerSeedRingHit->localPosition());
        GlobalPoint outerSeedHitGlobalPosition = tracker->idToDet(outerSeedRingHit->geographicalId())->surface().toGlobal(outerSeedRingHit->localPosition());
        
        // extrapolation parameters, phio: [0,2pi]
        double d0 = 0.0;
        double phi0 = -99.;
        double k0   = -99999999.99;
        
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
          }
        }
        
        // continue if valid extrapolation parameters have been found
        if ( (phi0 != -99.) && (k0 != -99999999.99) ) {
          Roads::const_iterator roadMapEntry = roads->getRoadSet(roadSeed);
          
          // create cloud
          RoadSearchCloud cloud;
          
          // seed edm::Ref
          RoadSearchCloud::SeedRef seedRef(input,seedCounter);
          
          cloud.addSeed(seedRef);
          
          unsigned int usedLayers = 0;
          
          for ( Roads::RoadSet::const_iterator roadSetVector = roadMapEntry->second.begin();
                roadSetVector != roadMapEntry->second.end();
                ++roadSetVector ) {
            
            unsigned int usedHitsInThisLayer = 0;
            
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
              
              // calculate range in phi around ringPhi
              double upperPhiRangeBorder = map_phi2(ringPhi + theMinimumHalfRoad);
              double lowerPhiRangeBorder = map_phi2(ringPhi - theMinimumHalfRoad);
              
              if ( lowerPhiRangeBorder <= upperPhiRangeBorder ) {
                
                for ( Ring::const_iterator detid = (*ring)->lower_bound(lowerPhiRangeBorder); detid != (*ring)->upper_bound(upperPhiRangeBorder); ++detid) {
                  
                  usedHitsInThisLayer += FillRecHitsIntoCloudGeneral(detid->second,d0,phi0,k0,roadType,ringPhi,&(*seed),
                                                                     tracker.product(),theHitMatcher,cloud);
                }
                
              } else {
                for ( Ring::const_iterator detid = (*ring)->lower_bound(lowerPhiRangeBorder); detid != (*ring)->end(); ++detid) {
                  usedHitsInThisLayer += FillRecHitsIntoCloudGeneral(detid->second,d0,phi0,k0,roadType,ringPhi,&(*seed),
                                                                     tracker.product(),theHitMatcher,cloud);
                }
                
                for ( Ring::const_iterator detid = (*ring)->begin(); detid != (*ring)->upper_bound(upperPhiRangeBorder); ++detid) {
                  usedHitsInThisLayer += FillRecHitsIntoCloudGeneral(detid->second,d0,phi0,k0,roadType,ringPhi,&(*seed),
                                                                     tracker.product(),theHitMatcher,cloud);
                  
                }
              }
              
            }
            
            if ( usedHitsInThisLayer > 0 ) {
              ++usedLayers;
            }
            
          }
          
          if ( usedLayers >= minNumberOfUsedLayersPerRoad &&
               (roadMapEntry->second.size() - usedLayers) <= maxNumberOfMissedLayersPerRoad ) {
            
            CloudArray[phibin][etabin].push_back(cloud);
            
            if ( roadType == Roads::RPhi ){ 
              output_ << "This r-phi seed yields a cloud with " <<cloud.size() <<" hits\n";
            } else {
              output_ << "This z-phi seed yields a cloud with "<<cloud.size() <<" hits\n";
            }
          } else {
            if ( roadType == Roads::RPhi ){ 
              output_ << "This r-phi seed yields no clouds\n";
            } else {
              output_ << "This z-phi seed yields no clouds\n";
            }
          }
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
  
  //   edm::LogInfo("RoadSearch") << output_.str();
  
  delete theHitMatcher;
  edm::LogInfo("RoadSearch") << "Found " << output.size() << " clouds."; 
  
}

unsigned int RoadSearchCloudMakerAlgorithm::FillRecHitsIntoCloudGeneral(DetId id, double d0, double phi0, double k0, Roads::type roadType, double ringPhi,
                                                                        const TrajectorySeed* seed, 
                                                                        // 									std::vector<bool> &usedLayersArray,
                                                                        // 									const std::vector<unsigned int> &numberOfLayersPerSubdetector,
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
      
      if ( roadType == Roads::RPhi ) {
        if (double_ring_layer && isSingleLayer(hitId)) {
          //
          //  This is where the barrel stereoRecHits end up for Roads::RPhi
          //
          
          // Adjust matched hit for track angle
          const SiStripMatchedRecHit2D *theRH = dynamic_cast<SiStripMatchedRecHit2D*>(*recHitIterator);
          //const GluedGeomDet *theGluedDet = dynamic_cast<const GluedGeomDet*>(tracker->idToDet(theRH->geographicalId()));
          const GeomDet *recHitGeomDet = tracker->idToDet(hitId);
          const GluedGeomDet *theGluedDet = dynamic_cast<const GluedGeomDet*>(recHitGeomDet);
          
          const GeomDetUnit* theMonoDet = theGluedDet->monoDet();
          const SiStripRecHit2D* theMonoHit   = theRH->monoHit(); 
          
          GlobalPoint gcenterofstrip=(theMonoDet->surface()).toGlobal(theMonoHit->localPosition());
          //GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
          //LocalVector ltrackdirection=(theMonoDet->surface()).toLocal(gtrackdirection);
          
          float gtrackangle_xy = map_phi2(phi0 + 2.0*asin(0.5*gcenterofstrip.perp()*k0));
          float rzangle = atan2(gcenterofstrip.perp(),gcenterofstrip.z());
          
          //GlobalVector gtrackangle(cos(gtrackangle_xy)*gcenterofstrip.perp(),
          //			   sin(gtrackangle_xy)*gcenterofstrip.perp(),
          //			   gcenterofstrip.z());
          //LocalVector trackdirection=((tracker->idToDet(hitId))->surface()).toLocal(gtrackangle);
          
          GlobalVector gtrackangle2(cos(gtrackangle_xy)*sin(rzangle),
                                    sin(gtrackangle_xy)*sin(rzangle),
                                    cos(rzangle));
          LocalVector trackdirection2=((tracker->idToDet(hitId))->surface()).toLocal(gtrackangle2);
          SiStripMatchedRecHit2D* theCorrectedHit = theHitMatcher->match(theRH,
                                                                         theGluedDet,
                                                                         trackdirection2);
          
          
          if (theCorrectedHit != 0){
            
            GlobalPoint ghit = tracker->idToDet(theCorrectedHit->geographicalId())->surface().toGlobal(theCorrectedHit->localPosition());	    
            double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
            double hitphi = map_phi(ghit.phi());
            double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
            
            float dp = (hitphi-phi);
            float dx = hitRadius*tan(dp);
            
            //if ( std::abs(hitphi-phi) < 6.0*phiMax(seed,roadType,phi0,k0) ) {
            // switch cut to dx instead of dphi
            if ( std::abs(dx) < phiMax(seed,roadType,phi0,k0) ) {
              if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
                //cloud.addHit((TrackingRecHit*)theCorrectedHit->clone());
                cloud.addHit((TrackingRecHit*)recHit->clone());
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
              
	    //if ( std::abs(hitphi-phi) < phiMax(seed,roadType,phi0,k0) ) {
	    // switch cut to dx instead of dphi
	    if ( std::abs(dx) < phiMax(seed,roadType,phi0,k0) ) {
              if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		cloud.addHit((TrackingRecHit*)recHit->clone());
		++usedRecHits;
	      }
	    }
	  } 
	  else {
	    LocalPoint hit = recHit->localPosition();
	    const StripTopology *topology = dynamic_cast<const StripTopology*>(&(tracker->idToDetUnit(hitId)->topology()));
	    double stripAngle = topology->stripAngle(topology->strip(hit));
	    double stripLength = topology->localStripLength(hit);
	    LocalPoint upperLocalBoundary(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	    LocalPoint lowerLocalBoundary(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	    double upperBoundaryRadius = tracker->idToDetUnit(hitId)->surface().toGlobal(upperLocalBoundary).perp(); 
	    double lowerBoundaryRadius = tracker->idToDetUnit(hitId)->surface().toGlobal(lowerLocalBoundary).perp();
	    double upperBoundaryPhi = phiFromExtrapolation(d0,phi0,k0,upperBoundaryRadius,roadType);
	    double lowerBoundaryPhi = phiFromExtrapolation(d0,phi0,k0,lowerBoundaryRadius,roadType);
	    double hitPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(hit).phi());
              
	    //double midpointRadius = 0.5*(upperBoundaryRadius+lowerBoundaryRadius);
	    //double midpointPhi = phiFromExtrapolation(d0,phi0,k0,midpointRadius,roadType);
	    //float dx = midpointRadius*tan(hitPhi-midpointPhi);
              
	    if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
	      //
	      //  This is where the disk (???) rphiRecHits end up for Roads::RPhi
	      //
	      if ( ((lowerBoundaryPhi - phiMax(seed,roadType,phi0,k0)) < hitPhi) &&
		   ((upperBoundaryPhi + phiMax(seed,roadType,phi0,k0)) > hitPhi) ) {
		if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		  cloud.addHit((TrackingRecHit*)recHit->clone());
		  ++usedRecHits;
		}
	      }
	    } else {
	      //
	      //  some type of hit (see above) gets here
	      //
	      if ( ((upperBoundaryPhi - phiMax(seed,roadType,phi0,k0)) < hitPhi) &&
		   ((lowerBoundaryPhi + phiMax(seed,roadType,phi0,k0)) > hitPhi) ) {
		if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		  cloud.addHit((TrackingRecHit*)recHit->clone());
		  ++usedRecHits;
		}
	      }
	    }
	  }
	} 
      } else {
	//
	// roadType == Roads::ZPhi
	//
	if (double_ring_layer && isSingleLayer(hitId)) {
	  double hitphi = map_phi(tracker->idToDet(hitId)->surface().toGlobal(recHit->localPosition()).phi());
	  //double hitRadius = tracker->idToDetUnit(id)->surface().toGlobal(recHit->localPosition()).perp();
	  double hitZ = tracker->idToDet(hitId)->surface().toGlobal(recHit->localPosition()).z();
	  double phi = phiFromExtrapolation(d0,phi0,k0,hitZ,roadType);
              
	  //float dp = hitphi-phi;
	  //float dx = hitZ*tan(dp);
              
	  //
	  //  This is where the disk stereoRecHits end up for Roads::ZPhi
	  //
	  if ( std::abs(hitphi-phi) < 6.0*phiMax(seed,roadType,phi0,k0) ) {
	    if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
	      cloud.addHit((TrackingRecHit*)recHit->clone());
	      ++usedRecHits;
	    }
	  }
	} else { // Single layer hits here
	  if ( isBarrelSensor(hitId) ) {
	    LocalPoint hit = recHit->localPosition();
	    const StripTopology *topology = dynamic_cast<const StripTopology*>(&(tracker->idToDetUnit(hitId)->topology()));
	    double stripLength = topology->stripLength();
	    LocalPoint upperLocalBoundary(hit.x(),hit.y()+stripLength/2,0);
	    LocalPoint lowerLocalBoundary(hit.x(),hit.y()-stripLength/2,0);
	    double upperBoundaryZ = tracker->idToDetUnit(hitId)->surface().toGlobal(upperLocalBoundary).z(); 
	    double lowerBoundaryZ = tracker->idToDetUnit(hitId)->surface().toGlobal(lowerLocalBoundary).z();
	    double upperBoundaryPhi = phiFromExtrapolation(d0,phi0,k0,upperBoundaryZ,roadType);
	    double lowerBoundaryPhi = phiFromExtrapolation(d0,phi0,k0,lowerBoundaryZ,roadType);
	    double hitPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(recHit->localPosition()).phi());
                
	    //double midpointZ = 0.5*(upperBoundaryZ+lowerBoundaryZ);
	    //double midpointPhi = phiFromExtrapolation(d0,phi0,k0,midpointZ,roadType);
	    //float dx = midpointZ*tan(hitPhi-midpointPhi);
                
	    if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
	      //
	      //  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
	      //
	      if ( ((lowerBoundaryPhi - phiMax(seed,roadType,phi0,k0)) < hitPhi) &&
		   ((upperBoundaryPhi + phiMax(seed,roadType,phi0,k0)) > hitPhi) ) {
		if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		  cloud.addHit((TrackingRecHit*)recHit->clone());
		  ++usedRecHits;
		}
	      }
	    } else {
	      //
	      //  This is where the barrel (???) rphiRecHits end up for Roads::ZPhi
	      //
	      if ( ((upperBoundaryPhi - phiMax(seed,roadType,phi0,k0)) < hitPhi) &&
		   ((lowerBoundaryPhi + phiMax(seed,roadType,phi0,k0)) > hitPhi) ) {
		if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		  cloud.addHit((TrackingRecHit*)recHit->clone());
		  ++usedRecHits;
		}
	      }
	    }
	  } else {
	    LocalPoint hit = recHit->localPosition();
	    const StripTopology *topology = dynamic_cast<const StripTopology*>(&(tracker->idToDetUnit(hitId)->topology()));
	    double stripAngle = topology->stripAngle(topology->strip(hit));
	    double stripLength = topology->localStripLength(hit);
	    LocalPoint upperLocalBoundary(hit.x()-stripLength/2*std::sin(stripAngle),hit.y()+stripLength/2*std::cos(stripAngle),0);
	    LocalPoint lowerLocalBoundary(hit.x()+stripLength/2*std::sin(stripAngle),hit.y()-stripLength/2*std::cos(stripAngle),0);
	    double diskZ = tracker->idToDetUnit(hitId)->surface().toGlobal(upperLocalBoundary).z(); 
	    double upperBoundaryPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(upperLocalBoundary).phi()); 
	    double lowerBoundaryPhi = map_phi(tracker->idToDetUnit(hitId)->surface().toGlobal(lowerLocalBoundary).phi());
	    double roadPhi =  phiFromExtrapolation(d0,phi0,k0,diskZ,roadType);
                
	    //double midpointPhi = 0.5*(upperBoundaryPhi+lowerBoundaryPhi);
	    //if (fabs(lowerBoundaryPhi-upperBoundaryPhi) > 3.14159
	    //  midpointPhi= 0.5*(upperBoundaryRadius-lowerBoundaryRadius);
	    //float dp = midpointPhi-roadPhi;
	    //float dx = diskZ*tan(dp);
	    if ( lowerBoundaryPhi <= upperBoundaryPhi ) {
	      //
	      //  This is where the disk rphiRecHits end up for Roads::ZPhi
	      //
	      if ( ((lowerBoundaryPhi - phiMax(seed,roadType,phi0,k0)) < roadPhi) &&
		   ((upperBoundaryPhi + phiMax(seed,roadType,phi0,k0)) > roadPhi) ) {
		if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		  cloud.addHit((TrackingRecHit*)recHit->clone());
		  ++usedRecHits;
		}
	      }
	    } else {
	      //
	      //  no hits (see above) seem to get here
	      //
	      if ( ((upperBoundaryPhi - phiMax(seed,roadType,phi0,k0)) < roadPhi) &&
		   ((lowerBoundaryPhi + phiMax(seed,roadType,phi0,k0)) > roadPhi) ) {
		if ((usedRecHits < maxDetHitsInCloudPerDetId) && (cloud.size() < maxRecHitsInCloud_)) {
		  cloud.addHit((TrackingRecHit*)recHit->clone());
		  ++usedRecHits;
		}
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
            
            //if ( std::abs(hitphi-phi) < phiMax(seed,roadType,phi0,k0) ) {
            // switch cut to dx instead of dphi
            if ( std::abs(dx) < phiMax(seed,roadType,phi0,k0) ) {
              cloud.addHit((TrackingRecHit*)recHit->clone());
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
              
	    if ( std::abs(hitphi-phi) < phiMax(seed,roadType,phi0,k0) ) {
	      cloud.addHit((TrackingRecHit*)recHit->clone());
	      ++usedRecHits;
	    }	
	  }
	} else {
            
	  GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
            
	  double  phi = phiFromExtrapolation(d0,phi0,k0,ghit.z(),roadType);
            
	  //double hitphi = map_phi(ghit.phi());
	  //float dx = ghit.z()*tan(hitphi-phi);
            
	  if ( (phi - phiMax(seed,roadType,phi0,k0)) < ringPhi && (phi + phiMax(seed,roadType,phi0,k0))>ringPhi ) {
	    cloud.addHit((TrackingRecHit*)recHit->clone());
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
                                                                    const TrajectorySeed* seed, 
                                                                    // 								    std::vector<bool> &usedLayersArray, 
                                                                    // 								    const std::vector<unsigned int> &numberOfLayersPerSubdetector,
                                                                    const TrackerGeometry *tracker, RoadSearchCloud &cloud) {
  
  
  unsigned int usedRecHits = 0;
  
  // Get Geometry
  //const RectangularPixelTopology *topology = dynamic_cast<const RectangularPixelTopology*>(&(tracker->idToDetUnit(id)->topology()));
  
  
  // retrieve vector<SiPixelRecHit> for id
  // loop over SiPixelRecHit
  // check if compatible with cloud, fill into cloud
  
  const SiPixelRecHitCollection::range recHitRange = inputRecHits->get(id);
  
  for ( SiPixelRecHitCollection::const_iterator recHitIterator = recHitRange.first; 
        recHitIterator != recHitRange.second; ++recHitIterator) {
    
    const SiPixelRecHit * recHit = &(*recHitIterator);
    
    if ( roadType == Roads::RPhi ) {
      
      if ( isBarrelSensor(id) ) {
        // Barrel Pixel, RoadType RPHI
        
        GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
        double hitRadius = sqrt(ghit.x()*ghit.x()+ghit.y()*ghit.y());
        double hitphi = map_phi(ghit.phi());
        double phi = phiFromExtrapolation(d0,phi0,k0,hitRadius,roadType);
        
        if ( std::abs(hitphi-phi) < phiMax(seed,roadType,phi0,k0) ) {
          cloud.addHit((TrackingRecHit*)recHit->clone());
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
        
        if ( std::abs(hitphi-phi) < phiMax(seed,roadType,phi0,k0) ) {
          cloud.addHit((TrackingRecHit*)recHit->clone());
          ++usedRecHits;
        }	
      }
    } 
    
    else {
      
      GlobalPoint ghit = tracker->idToDet(recHit->geographicalId())->surface().toGlobal(recHit->localPosition());
      
      double  phi = phiFromExtrapolation(d0,phi0,k0,ghit.z(),roadType);
      if ( (phi - phiMax(seed,roadType,phi0,k0)) < ringPhi && (phi + phiMax(seed,roadType,phi0,k0))>ringPhi ) {
        cloud.addHit((TrackingRecHit*)recHit->clone());
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

double RoadSearchCloudMakerAlgorithm::phiMax(const TrajectorySeed *seed, Roads::type roadType,
                                             double phi0, double k0) {
  
  double dphi;
  if ( roadType == Roads::RPhi ) {
    //dphi = theRPhiRoadSize + fabs(k0);
    // switch cut to dx instead of dphi
    // Still call this dphi, but will now be dx
    dphi = theRPhiRoadSize + 0.15*82.0*fabs(k0);
  }
  else if ( roadType == Roads::ZPhi ) {
    dphi = theZPhiRoadSize;
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

RoadSearchCloudCollection RoadSearchCloudMakerAlgorithm::Clean(RoadSearchCloudCollection* inputCollection){
  
  RoadSearchCloudCollection output;
  
  //
  //  no raw clouds - nothing to try merging
  //
  
  if ( inputCollection->empty() ){
    LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
    return output;  
  }
  
  //
  //  1 raw cloud - nothing to try merging, but one cloud to duplicate
  //
  
  if ( 1==inputCollection->size() ){
    RoadSearchCloud *temp = inputCollection->begin()->clone();
    output.push_back(*temp);
    delete temp;
    LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
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
    LogDebug("RoadSearch") << "number of ref in rawcloud " << raw_cloud->seeds().size(); 
    
    // produce output cloud where other clouds are merged in
    // create temp pointer for clone which will be deleted afterwards
    RoadSearchCloud *temp_lone_cloud = raw_cloud->clone();
    RoadSearchCloud lone_cloud = *temp_lone_cloud;
    int second_cloud_ctr=raw_cloud_ctr;
    for ( RoadSearchCloudCollection::const_iterator second_cloud = raw_cloud+1; second_cloud != inputCollection->end(); ++second_cloud) {
      second_cloud_ctr++;
      
      std::vector<const TrackingRecHit*> unshared_hits;
      
      if ( already_gone[second_cloud_ctr-1] )continue;
      LogDebug("RoadSearch") << "number of ref in second_cloud " << second_cloud->seeds().size(); 
      
      for ( RoadSearchCloud::RecHitOwnVector::const_iterator second_cloud_hit = second_cloud->begin_hits();
            second_cloud_hit != second_cloud->end_hits();
            ++ second_cloud_hit ) {
        bool is_shared = false;
        for ( RoadSearchCloud::RecHitOwnVector::const_iterator lone_cloud_hit = lone_cloud.begin_hits();
              lone_cloud_hit != lone_cloud.end_hits();
              ++ lone_cloud_hit ) {
          
          if (lone_cloud_hit->geographicalId() == second_cloud_hit->geographicalId())
            if (lone_cloud_hit->localPosition().x() == second_cloud_hit->localPosition().x())
              if (lone_cloud_hit->localPosition().y() == second_cloud_hit->localPosition().y())
		{is_shared=true; break;}
        }
	if (!is_shared)  unshared_hits.push_back(&(*second_cloud_hit));
          
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
        
      float f_lone_shared=float(second_cloud->size()-unshared_hits.size())/float(lone_cloud.size());
      float f_second_shared=float(second_cloud->size()-unshared_hits.size())/float(second_cloud->size());
        
      if ( ( (f_lone_shared > mergingFraction_)||(f_second_shared > mergingFraction_) ) 
	   && (lone_cloud.size()+unshared_hits.size() <= maxRecHitsInCloud_) ){
          
	LogDebug("RoadSearch") << " Merge CloudA: " << raw_cloud_ctr << " with  CloudB: " << second_cloud_ctr 
			       << " Shared fractions are " << f_lone_shared << " and " << f_second_shared;
          
	//
	//  got a cloud to merge
	//
	for (unsigned int k=0; k<unshared_hits.size(); ++k) {
	  lone_cloud.addHit(unshared_hits[k]->clone());
	}
          
	// add seed of second_cloud to lone_cloud
	for ( RoadSearchCloud::SeedRefs::const_iterator secondseedref = second_cloud->begin_seeds();
	      secondseedref != second_cloud->end_seeds();
	      ++secondseedref ) {
	  lone_cloud.addSeed(*secondseedref);
	}
          
	already_gone[second_cloud_ctr-1]=true;
          
      }//end got a cloud to merge
        
    }//interate over all second clouds
      
    LogDebug("RoadSearch") << "number of ref in cloud " << lone_cloud.seeds().size(); 
      
    output.push_back(lone_cloud);
    // delete temp_lone_cloud pointer
    delete temp_lone_cloud;
      
  }//iterate over all raw clouds
    
  LogDebug("RoadSearch") << "Found " << output.size() << " clean clouds.";
    
  return output;
}

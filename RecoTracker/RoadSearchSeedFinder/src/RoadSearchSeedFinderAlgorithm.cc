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
// $Author: noeding $
// $Date: 2006/08/12 00:30:17 $
// $Revision: 1.13 $
//

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinderAlgorithm.h"

#include "FWCore/Framework/interface/Handle.h"
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

#include "RecoTracker/RoadMapRecord/interface/Roads.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

RoadSearchSeedFinderAlgorithm::RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
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

  const std::vector<DetId> availableIDs = matchedRecHits->ids();
  const std::vector<DetId> availableIDs2 = rphiRecHits->ids();
  const std::vector<DetId> availableIDs3 = stereoRecHits->ids();

  //for (int i=0;i<(int) availableIDs.size();++i) {
  //edm::LogError("RoadSearch") << "ID matched " << availableIDs[i].rawId();
  //}

  //for (int i=0;i<(int) availableIDs2.size();++i) {
  //edm::LogError("RoadSearch") << "ID " << availableIDs2[i].rawId();
   //if ( (unsigned int)availableIDs2[i].subdetId() == StripSubdetector::TIB ) {
    //TIBDetId tibid(availableIDs2[i].rawId()); 
    //if (tibid.glued())     edm::LogError("RoadSearch") << "ID " << availableIDs2[i].rawId();
    //}
  //} 
 
  //for (int i=0;i<(int) availableIDs3.size();++i) {
  //edm::LogError("RoadSearch") << "ID stereo " << availableIDs3[i].rawId();
  //} 

  // get roads
  edm::ESHandle<Roads> roads;
  es.get<TrackerDigiGeometryRecord>().get(roads);

  // calculate maximal possible delta phi for given delta r and parameter pTmin
  double ptmin = conf_.getParameter<double>("MinimalReconstructedTransverseMomentum");

  // correction for B given in T, delta r given in cm, ptmin given in GeV/c
  double speedOfLight = 2.99792458e8;
  double unitCorrection = speedOfLight * 1e-2 * 1e-9;

  // B in T, right now hardcoded, has to come from magnetic field service
  double B = 4.0;

  // initialize general hit access for road search
  DetHitAccess innerSeedHitVector(rphiRecHits,stereoRecHits,matchedRecHits,pixelRecHits);
  DetHitAccess outerSeedHitVector(rphiRecHits,stereoRecHits,matchedRecHits,pixelRecHits);

   // loop over seed Ring pairs
  for ( Roads::const_iterator road = roads->begin(); road != roads->end(); ++road ) {

    Roads::RoadSeed seed = (*road).first;
    //edm::LogError("RoadSearch") << "ROAD SEEDS: " << seed.first.getindex() << " " << seed.second.getindex();
    // loop over detid's in seed rings
    for ( Ring::const_iterator innerRingDetId = seed.first.begin(); innerRingDetId != seed.first.end(); ++innerRingDetId ) {

      //uint32_t detId_rawid = innerRingDetId->second.rawId();
      //DetId detId_tmp(detId_rawid);

      //uint32_t detId_tmp = innerRingDetId->second.rawId();
      StripSubdetector StripDetId(innerRingDetId->second);
      DetId tmp(StripDetId.glued());
      //edm::LogError("RoadSearch") << "try to fine detid: " << tmp.rawId() << " oldid: " << innerRingDetId->second.rawId();


      //for (int i=0;i<availableIDs.size();++i) {
      //edm::LogError("RoadSearch") << "ID " << availableIDs[i].rawId();
      //}

      //if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),innerRingDetId->second.glued()) ) {
      if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),tmp) ) {
      //if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),innerRingDetId->second) ) {
	//edm::LogError("RoadSearch") << " cmp: " << innerRingDetId->second.rawId() << "  " << tmp.rawId();
      
      //for (int i=0; i<availableIDs.size(); ++i) {
      //edm::LogError("RoadSearch") << " available ID: " << availableIDs[i].rawId();
      //}
      
	//if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detId_tmp) ) 
	//edm::LogError("RoadSearch") << "WRONG!!!";

	//if ( availableIDs.end() != std::find(availableIDs.begin(),availableIDs.end(),detId_tmp) ) {

	for ( Ring::const_iterator outerRingDetId = seed.second.begin(); outerRingDetId != seed.second.end(); ++outerRingDetId ) {

	  StripSubdetector OuterStripDetId(outerRingDetId->second);
	  uint32_t test=OuterStripDetId.rawId();
	  if (OuterStripDetId.glued()!=0) {
	    test =  OuterStripDetId.glued();
	  }
	  //DetId out_tmp(OuterStripDetId.glued());
	  //DetId out_tmp(test);
	  if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),outerRingDetId->second) ) {
	  //if ( availableIDs2.end() != std::find(availableIDs2.begin(),availableIDs2.end(),out_tmp) ) {

	    	        
	    // loop over outer dethits
	    //edm::LogError("RoadSearch") << "outerSeedDetHits... " ;
	    //edm::OwnVector<TrackingRecHit> outerSeedDetHits = outerSeedHitVector.getHitVector(&(outerRingDetId->second));
	    //for (edm::OwnVector<TrackingRecHit>::const_iterator outerSeedDetHit = outerSeedDetHits.begin();
	    // outerSeedDetHit != outerSeedDetHits.end(); ++outerSeedDetHit) {

	    //edm::LogError("RoadSearch") << "innerSeedDetHits: "  << seed.first.print() ;
	    edm::OwnVector<TrackingRecHit> innerSeedDetHits = innerSeedHitVector.getHitVector(&(innerRingDetId->second));
	    
	    // loop over inner dethits
	    for (edm::OwnVector<TrackingRecHit>::const_iterator innerSeedDetHit = innerSeedDetHits.begin();
		 innerSeedDetHit != innerSeedDetHits.end(); ++innerSeedDetHit) {

	      //edm::LogError("RoadSearch") << "outerSeedDetHits: " << seed.second.print() ;
	      edm::OwnVector<TrackingRecHit> outerSeedDetHits = outerSeedHitVector.getHitVector(&(outerRingDetId->second));
	        
	      //loop over outer dethits
	      for (edm::OwnVector<TrackingRecHit>::const_iterator outerSeedDetHit = outerSeedDetHits.begin();
		   outerSeedDetHit != outerSeedDetHits.end(); ++outerSeedDetHit) {

		// get tracker geometry
		edm::ESHandle<TrackerGeometry> tracker;
		es.get<TrackerDigiGeometryRecord>().get(tracker);

		GlobalPoint inner = tracker->idToDet(innerSeedDetHit->geographicalId())->surface().toGlobal(innerSeedDetHit->localPosition());
		GlobalPoint outer = tracker->idToDet(outerSeedDetHit->geographicalId())->surface().toGlobal(outerSeedDetHit->localPosition());

		// calculate deltaPhi in [0,2pi]
		double deltaPhi = std::abs(inner.phi() - outer.phi());
		double pi = 3.14159265358979312;
		if ( deltaPhi < 0 ) deltaPhi = 2*pi - deltaPhi;
	    
		double innerr = std::sqrt(inner.x()*inner.x()+inner.y()*inner.y());
		double outerr = std::sqrt(outer.x()*outer.x()+outer.y()*outer.y());

		// calculate maximal delta phi in [0,2pi]
		double deltaPhiMax = std::abs( std::asin(unitCorrection * B * innerr / ptmin) - std::asin(unitCorrection * B * outerr / ptmin) );
		if ( deltaPhiMax < 0 ) deltaPhiMax = 2*pi - deltaPhiMax;

		if ( deltaPhi <= deltaPhiMax ) {
	      
		  // use correct tk seed generator from consecutive hits
		  // make PTrajectoryOnState from two hits and region (beamspot)
		  GlobalTrackingRegion region;
		  GlobalError vtxerr( std::sqrt(region.originRBound()),
				      0, std::sqrt(region.originRBound()),
				      0, 0, std::sqrt(region.originZBound()));

		  double x0=0.0,y0=0.0,z0=0.0;
                  bool nofieldcosmic = conf_.getParameter<bool>("StraightLineNoBeamSpotSeed");
                  if (nofieldcosmic){
                    double phi0=atan2(outer.y()-inner.y(),outer.x()-inner.x());
                    double alpha=atan2(inner.y(),inner.x());
                    double d1=sqrt(inner.x()*inner.x()+inner.y()*inner.y());
                    double d0=-d1*sin(alpha-phi0); x0=d0*sin(phi0); y0=-d0*cos(phi0);
                    double l1=0.0,l2=0.0;
                    if (fabs(cos(phi0))>0.1){
                      l1=(inner.x()-x0)/cos(phi0);l2=(outer.x()-x0)/cos(phi0);  
                    }else{
                      l1=(inner.y()-y0)/sin(phi0);l2=(outer.y()-y0)/sin(phi0);  
                    }
                    z0=(l2*inner.z()-l1*outer.z())/(l2-l1);
//                    std::cout << "In RSSF, d0,phi0,x0,y0,z0 " << d0 << " " << phi0 << " " << x0 << " " << y0 << " " << z0 << std::endl;
                  }

		  FastHelix helix(outer, inner, GlobalPoint(x0,y0,z0),es);

                  FreeTrajectoryState fts( helix.stateAtVertex().parameters(),
                                           initialError( &(*outerSeedDetHit), &(*innerSeedDetHit),
                                                         region.origin(), vtxerr));

                  edm::ESHandle<MagneticField> pSetup;
                  es.get<IdealMagneticFieldRecord>().get(pSetup);

                  AnalyticalPropagator  thePropagator(&(*pSetup), alongMomentum);

                  KFUpdator theUpdator;

                  const TrajectoryStateOnSurface innerState = thePropagator.propagate(fts,tracker->idToDet(innerSeedDetHit->geographicalId())->surface());

		  if (innerState.isValid()){
		    //
		    // create the OwnVector of TrackingRecHits
		    //
		    edm::OwnVector<TrackingRecHit> rh;

		    //
		    // memory leak??? TB
		    //
		    rh.push_back(innerSeedDetHit->clone());
		    rh.push_back(outerSeedDetHit->clone());
		    TrajectoryStateTransform transformer;
		    
		    PTrajectoryStateOnDet * PTraj=  transformer.persistentState(innerState, innerSeedDetHit->geographicalId().rawId());
		    TrajectorySeed ts(*PTraj,rh,alongMomentum);
		  
		    // 060811/OLI: memory leak fix as suggested by Chris
                    delete PTraj;  

		    // add seed to collection
		    output.push_back(ts);

		    //edm::LogError("RoadSearch") << "innerSeedDetHits: "  << innerRingDetId->second.rawId() << "; " <<seed.first.print() ;
		    //edm::LogError("RoadSearch") << "outerSeedDetHits: " << outerRingDetId->second.rawId() << "; " << seed.second.print() ;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  edm::LogInfo("RoadSearch") << "Found " << output.size() << " seeds."; 

};

CurvilinearTrajectoryError RoadSearchSeedFinderAlgorithm::
initialError( const TrackingRecHit* outerHit,
              const TrackingRecHit* innerHit,
              const GlobalPoint& vertexPos,
              const GlobalError& vertexErr)
{
  AlgebraicSymMatrix C(5,1);

  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr;

  return CurvilinearTrajectoryError(C);
}

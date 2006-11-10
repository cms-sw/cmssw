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
// $Date: 2006/09/08 19:26:19 $
// $Revision: 1.19 $
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

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

const double speedOfLight = 2.99792458e8;
const double unitCorrection = speedOfLight * 1e-2 * 1e-9;

RoadSearchSeedFinderAlgorithm::RoadSearchSeedFinderAlgorithm(const edm::ParameterSet& conf) { 

  NoFieldCosmic_ = conf.getParameter<bool>("StraightLineNoBeamSpotSeed");
  theMinPt_ = conf.getParameter<double>("MinimalReconstructedTransverseMomentum");

  // configure DetHitAccess
  innerSeedHitVector_.setMode(DetHitAccess::rphi);
  outerSeedHitVector_.setMode(DetHitAccess::rphi);

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
  innerSeedHitVector_.setMode(DetHitAccess::rphi);
  outerSeedHitVector_.setMode(DetHitAccess::rphi);

  // get roads
  edm::ESHandle<Roads> roads;
  es.get<TrackerDigiGeometryRecord>().get(roads);
  roads_ = roads.product();

  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  tracker_ = tracker.product();

  // get magnetic field
  edm::ESHandle<MagneticField> magnet;
  es.get<IdealMagneticFieldRecord>().get(magnet);
  magnet_ = magnet.product();

  // loop over seed Ring pairs
  for ( Roads::const_iterator road = roads_->begin(); road != roads_->end(); ++road ) {

    Roads::RoadSeed seed = (*road).first;
  
    // loop over detid's in seed rings
    for ( Ring::const_iterator innerRingDetId = seed.first.begin(); innerRingDetId != seed.first.end(); ++innerRingDetId ) {

      std::vector<TrackingRecHit*> innerSeedDetHits = innerSeedHitVector_.getHitVector(&(innerRingDetId->second));

      // loop over inner dethits
      for (std::vector<TrackingRecHit*>::const_iterator innerSeedDetHit = innerSeedDetHits.begin();
	   innerSeedDetHit != innerSeedDetHits.end(); ++innerSeedDetHit) {
	  
	GlobalPoint inner = tracker_->idToDet((*innerSeedDetHit)->geographicalId())->surface().toGlobal((*innerSeedDetHit)->localPosition());

	double innerphi = inner.phi();
	double upperPhiRangeBorder = innerphi + (1.0);
	double lowerPhiRangeBorder = innerphi - (1.0);
	if (upperPhiRangeBorder>Geom::twoPi()) upperPhiRangeBorder -= Geom::twoPi();
	if (lowerPhiRangeBorder<0.0) lowerPhiRangeBorder += Geom::twoPi();

	if (lowerPhiRangeBorder <= upperPhiRangeBorder ) {
	  for ( Ring::const_iterator outerRingDetId = seed.second.lower_bound(lowerPhiRangeBorder); 
		outerRingDetId != seed.second.upper_bound(upperPhiRangeBorder);
		++outerRingDetId) {
	    std::vector<TrackingRecHit*> outerSeedDetHits = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	    if ( outerSeedDetHits.size() > 0 ) {
	      makeSeedsFromInnerHit(&output,*innerSeedDetHit,&outerSeedDetHits,es);
	    }
	  }
	} else {
	  for ( Ring::const_iterator outerRingDetId = seed.second.begin(); 
		outerRingDetId != seed.second.upper_bound(upperPhiRangeBorder);
		++outerRingDetId) {
	    std::vector<TrackingRecHit*> outerSeedDetHits = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	    if ( outerSeedDetHits.size() > 0 ) {
	      makeSeedsFromInnerHit(&output,*innerSeedDetHit,&outerSeedDetHits,es);
	    }
	  }
	  for ( Ring::const_iterator outerRingDetId = seed.second.lower_bound(lowerPhiRangeBorder); 
		outerRingDetId != seed.second.end();
		++outerRingDetId) {
	    std::vector<TrackingRecHit*> outerSeedDetHits = outerSeedHitVector_.getHitVector(&(outerRingDetId->second));
	    if ( outerSeedDetHits.size() > 0 ) {
	      makeSeedsFromInnerHit(&output,*innerSeedDetHit,&outerSeedDetHits,es);
	    }
	  }
	}
      }
    }
  }

  LogDebug("RoadSearch") << "Found " << output.size() << " seeds."; 

}

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

TrajectorySeed RoadSearchSeedFinderAlgorithm::
makeSeedFromPair(const TrackingRecHit* innerSeedDetHit,
		 const GlobalPoint* inner,
		 const TrackingRecHit* outerSeedDetHit,
		 const GlobalPoint* outer,
		 const edm::EventSetup& es)
{


 // use correct tk seed generator from consecutive hits
  // make PTrajectoryOnState from two hits and region (beamspot)
  GlobalTrackingRegion region;
  GlobalError vtxerr( std::sqrt(region.originRBound()),
		      0, std::sqrt(region.originRBound()),
		      0, 0, std::sqrt(region.originZBound()));
  
  double x0=0.0,y0=0.0,z0=0.0;
  if (NoFieldCosmic_){
    double phi0=atan2(outer->y()-inner->y(),outer->x()-inner->x());
    double alpha=atan2(inner->y(),inner->x());
    double d1=sqrt(inner->x()*inner->x()+inner->y()*inner->y());
    double d0=-d1*sin(alpha-phi0); x0=d0*sin(phi0); y0=-d0*cos(phi0);
    double l1=0.0,l2=0.0;
    if (fabs(cos(phi0))>0.1){
      l1=(inner->x()-x0)/cos(phi0);l2=(outer->x()-x0)/cos(phi0);  
    }else{
      l1=(inner->y()-y0)/sin(phi0);l2=(outer->y()-y0)/sin(phi0);  
    }
    z0=(l2*inner->z()-l1*outer->z())/(l2-l1);
  }
  
  FastHelix helix(*outer, *inner, GlobalPoint(x0,y0,z0),es);
  
  FreeTrajectoryState fts( helix.stateAtVertex().parameters(),
			   initialError( &(*outerSeedDetHit), &(*innerSeedDetHit),
					 region.origin(), vtxerr));
  
  AnalyticalPropagator  thePropagator(magnet_, alongMomentum);
  
  KFUpdator theUpdator;
  
  const TrajectoryStateOnSurface innerState = thePropagator.propagate(fts,tracker_->idToDet(innerSeedDetHit->geographicalId())->surface());
  
  
  if (innerState.isValid()){
    //
    // create the OwnVector of TrackingRecHits
    //
    edm::OwnVector<TrackingRecHit> rh;
    
    rh.push_back(innerSeedDetHit->clone());
    rh.push_back(outerSeedDetHit->clone());
    TrajectoryStateTransform transformer;
    
    PTrajectoryStateOnDet * PTraj=  transformer.persistentState(innerState, innerSeedDetHit->geographicalId().rawId());
    TrajectorySeed ts(*PTraj,rh,alongMomentum);
    delete PTraj;  
    
    // return the seed
    return ts;
    
  }
  
  TrajectorySeed emptySeed;
  return emptySeed;
}



void  RoadSearchSeedFinderAlgorithm::
makeSeedsFromInnerHit(TrajectorySeedCollection* outputCollection,
		      const TrackingRecHit* innerSeedDetHit,
		      const std::vector<TrackingRecHit*>* outerSeedDetHits,
		      const edm::EventSetup& es)
{

  // calculate maximal possible delta phi for given delta r and parameter pTmin

  // use correct tk seed generator from consecutive hits
  // make PTrajectoryOnState from two hits and region (beamspot)
  GlobalTrackingRegion region;
  GlobalError vtxerr( std::sqrt(region.originRBound()),
		      0, std::sqrt(region.originRBound()),
		      0, 0, std::sqrt(region.originZBound()));
  
  //loop over outer dethits
  for (std::vector<TrackingRecHit*>::const_iterator recHit_iter = outerSeedDetHits->begin();
       recHit_iter != outerSeedDetHits->end(); ++recHit_iter) {
    
    TrackingRecHit *outerSeedDetHit = (*recHit_iter);
    GlobalPoint inner = tracker_->idToDet(innerSeedDetHit->geographicalId())->surface().toGlobal(innerSeedDetHit->localPosition());
    GlobalPoint outer = tracker_->idToDet(outerSeedDetHit->geographicalId())->surface().toGlobal(outerSeedDetHit->localPosition());
    
    // calculate deltaPhi in [0,2pi]
    double deltaPhi = std::abs(inner.phi() - outer.phi());
    if ( deltaPhi < 0 ) deltaPhi = Geom::twoPi() - deltaPhi;
    
    double innerr = std::sqrt(inner.x()*inner.x()+inner.y()*inner.y());
    double outerr = std::sqrt(outer.x()*outer.x()+outer.y()*outer.y());
    
    // calculate maximal delta phi in [0,2pi]
    // use z component of magnetic field at inner and outer hit
    double deltaPhiMax = std::abs( std::asin(unitCorrection * magnet_->inTesla(inner).z() * innerr / theMinPt_) - std::asin(unitCorrection * magnet_->inTesla(outer).z() * outerr / theMinPt_) );
    if ( deltaPhiMax < 0 ) deltaPhiMax = Geom::twoPi() - deltaPhiMax;
    
    if ( deltaPhi <= deltaPhiMax ) {
      
      double x0=0.0,y0=0.0,z0=0.0;
      if (NoFieldCosmic_){
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
      }
      
      FastHelix helix(outer, inner, GlobalPoint(x0,y0,z0),es);
      
      FreeTrajectoryState fts( helix.stateAtVertex().parameters(),
			       initialError( &(*outerSeedDetHit), &(*innerSeedDetHit),
					     region.origin(), vtxerr));
      
      AnalyticalPropagator  thePropagator(magnet_, alongMomentum);
       
      const TrajectoryStateOnSurface innerState = thePropagator.propagate(fts,tracker_->idToDet(innerSeedDetHit->geographicalId())->surface());
      
      if (innerState.isValid()){
	//
	// create the OwnVector of TrackingRecHits
	edm::OwnVector<TrackingRecHit> rh;
	

	rh.push_back(innerSeedDetHit->clone());
	rh.push_back(outerSeedDetHit->clone());
	TrajectoryStateTransform transformer;
	
	PTrajectoryStateOnDet * PTraj=  transformer.persistentState(innerState, innerSeedDetHit->geographicalId().rawId());
	TrajectorySeed ts(*PTraj,rh,alongMomentum);
	delete PTraj;  
	
	// return the seed
	outputCollection->push_back(ts);
	
      } // InnerState is Valid
      
    }// Pair passes delta phi cut
    
  } // End loop over Outer Seed Hits
  
  //return SColl;
  
}


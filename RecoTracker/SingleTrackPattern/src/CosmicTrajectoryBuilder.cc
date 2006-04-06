//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia
#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"
CosmicTrajectoryBuilder::CosmicTrajectoryBuilder(const edm::ParameterSet& conf) : conf_(conf) { 
  //minimum number of hits per tracks
  theMinHits=conf_.getParameter<int>("MinHits");
  //cut on chi2
  chi2cut=conf_.getParameter<double>("Chi2Cut");
}


CosmicTrajectoryBuilder::~CosmicTrajectoryBuilder() {
}

void CosmicTrajectoryBuilder::init(const edm::EventSetup& es){


  //services
  es.get<IdealMagneticFieldRecord>().get(magfield);
  es.get<TrackerRecoGeometryRecord>().get( track );
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  //geometry
  bl=track->barrelLayers(); 

  //trackingtools
  thePropagator=    new AnalyticalPropagator(&(*magfield), anyDirection);
  theUpdator=       new KFUpdator();
  theEstimator=     new Chi2MeasurementEstimator(chi2cut);
  RHBuilder=        new TkTransientTrackingRecHitBuilder(&(*tracker));
  theFitter=        new KFTrajectoryFitter(*thePropagator,
					   *theUpdator,	
					   *theEstimator) ;

}

void CosmicTrajectoryBuilder::run(const TrajectorySeedCollection &collseed,
				  const SiStripRecHit2DLocalPosCollection &collstereo,
				  const SiStripRecHit2DLocalPosCollection &collrphi ,
				  const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const edm::EventSetup& es,
				  TrackCandidateCollection &output)
{
  //order all the hits
  vector<const TrackingRecHit*> allHits= SortHits(collstereo,collrphi,collmatched,collpixel);
  //create vector of transientrechit
  vector<const TrackingRecHit*>::iterator itrack;
  edm::OwnVector<TransientTrackingRecHit> hits;
  for(itrack=allHits.begin();itrack!=allHits.end();itrack++){
    //   GlobalPoint pippo =(&(*tracker))->idToDet((*itrack)->geographicalId())->surface().toGlobal((*itrack)->localPosition());
 
  
    hits.push_back(RHBuilder->build((*itrack) ));
  }

  std::vector<Trajectory> trajVec;
  std::vector<Trajectory>::iterator trajIter;
  // reco::Track * theTrack;

  TrajectorySeedCollection::const_iterator iseed;
  for(iseed=collseed.begin();iseed!=collseed.end();iseed++){
    //trajectory seed
    Trajectory startingTraj = createStartingTrajectory(*iseed);
  
    TSOS startingState=startingTSOS(*iseed);
   

    // AddHit(startingTraj,hits);
 
    // if ( qualityFilter( startingTraj)){
    //  trajVec = theFitter->fit(startingTraj);
    // }


    trajVec = theFitter->fit((*iseed), hits,startingState);
  
    
  }
 
};

Trajectory CosmicTrajectoryBuilder::createStartingTrajectory( const TrajectorySeed& seed) const
{
 
  Trajectory result( seed, seed.direction());

  std::vector<TM> seedMeas = seedMeasurements(seed);
  if ( !seedMeas.empty()) {
    for (std::vector<TM>::const_iterator i=seedMeas.begin(); i!=seedMeas.end(); i++){
      result.push(*i);
    }
  }
 
  return result;
}


std::vector<TrajectoryMeasurement> 
CosmicTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;

  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
 
    TransientTrackingRecHit* recHit = RHBuilder->build(&(*ihit));
    const GeomDet* hitGeomDet = (&(*tracker))->idToDet( ihit->geographicalId());
    TSOS invalidState( new BasicSingleTrajectoryState( hitGeomDet->surface()));

    if (ihit == hitRange.second - 1) {
      
      TSOS  updatedState=startingTSOS(seed);

      result.push_back(TM( invalidState, updatedState, recHit));
    } 
    else {
      result.push_back(TM( invalidState, recHit));
    }
    
  }

  return result;
};





vector<const TrackingRecHit*> 
CosmicTrajectoryBuilder::SortHits(const SiStripRecHit2DLocalPosCollection &collstereo,
				  const SiStripRecHit2DLocalPosCollection &collrphi ,
				  const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel){

  vector<const TrackingRecHit*> allHits;
  SiPixelRecHitCollection::const_iterator ipix;
  for(ipix=collpixel.begin();ipix!=collpixel.end();ipix++){
    allHits.push_back(&(*ipix));
  }

  SiStripRecHit2DLocalPosCollection::const_iterator istrip;
  for(istrip=collrphi.begin();istrip!=collrphi.end();istrip++){
    // bool differenthit= true;
   //   for (TrajectorySeed::const_iterator ihit = hitRange.first; 
   //	ihit != hitRange.second; ihit++) {
     //     if((*ihit).geographicalId()==(*istrip).geographicalId()) {
     //        if(((*ihit).localPosition()-(*istrip).localPosition()).mag()<0.1)  differenthit=false;
     //      }
     //    }
     //    if (differenthit)   
     allHits.push_back(&(*istrip)); 
  }

  for(istrip=collstereo.begin();istrip!=collstereo.end();istrip++){
    allHits.push_back(&(*istrip));
  }
  
  SiStripRecHit2DMatchedLocalPosCollection::const_iterator istripm;
  for(istripm=collmatched.begin();istripm!=collmatched.end();istripm++){
    
  }
  stable_sort(allHits.begin(),allHits.end(),CompareHitY(*tracker));

  return allHits;
};

TrajectoryStateOnSurface
CosmicTrajectoryBuilder::startingTSOS(const TrajectorySeed& seed)const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = (&(*tracker))->idToDet(DetId(pState.detId()));
  

  TSOS  State= tsTransform.transientState( pState, &(gdet->surface()), 
					   &(*magfield));
  return State;
}

void CosmicTrajectoryBuilder::AddHit(Trajectory traj,
				     edm::OwnVector<TransientTrackingRecHit> hits){


  TSOS currentState( traj.lastMeasurement().updatedState());
  bl=track->barrelLayers();

 
  unsigned int icosmhit=0;
  unsigned int indexlayer;
  vector<TM> meas;
  while ( icosmhit<hits.size()+1) {
    //   if ( icosmhit<hits.size()){
      //find the indexlayer
    DetId detid= hits[icosmhit].geographicalId();
    unsigned int subid=detid.subdetId();
    if    (subid==  PixelSubdetector::PixelBarrel){
      indexlayer=PXBDetId(detid).layer()-1;     
      const  PixelBarrelLayer *pix_lay=dynamic_cast<PixelBarrelLayer*>(bl[indexlayer]);
      meas=theLayerMeasurements->measurements(*pix_lay,currentState, 
					      *thePropagator, 
					      *theEstimator);
    }

    if    (subid==  StripSubdetector::TIB){
      indexlayer=TIBDetId(detid).layer()+3;
      const  TIBLayer *tib_lay=dynamic_cast<TIBLayer*>(bl[indexlayer]);
      meas=theLayerMeasurements->measurements(*tib_lay,currentState, 
					      *thePropagator, 
					      *theEstimator);
    }


    if    (subid== StripSubdetector::TOB){
      indexlayer=TOBDetId(detid).layer()+7;
    
      const  TOBLayer *tob_lay=dynamic_cast<TOBLayer*>(bl[indexlayer]);
      meas=theLayerMeasurements->measurements(*tob_lay,currentState, 
					      *thePropagator, 
					      *theEstimator);
    }

    for( vector<TM>::const_iterator itm = meas.begin(); 
	 itm != meas.end(); itm++) {
      updateTrajectory( traj, *itm);
    }
    
    
    icosmhit++;
  }

}

void CosmicTrajectoryBuilder::updateTrajectory( Trajectory& traj,
						       const TM& tm) const
{
  TSOS predictedState = tm.predictedState();
  const TransientTrackingRecHit *hit = tm.recHit();
 
  if ( hit->isValid()) {
    //  cout<<"VERO"<<endl;
    traj.push( TM( predictedState, theUpdator->update( predictedState, *hit),
		   hit, tm.estimate()));
  }
  else {
    //   cout<<"FALSO"<<endl;
    traj.push( TM( predictedState, &(*hit)));
  }
}
bool 
CosmicTrajectoryBuilder::qualityFilter(Trajectory traj){
  if ( traj.foundHits() >= theMinHits) {
    return true;
  }
  else {
    return false;
  }
}

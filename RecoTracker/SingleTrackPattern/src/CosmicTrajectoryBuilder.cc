//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia
#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
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
  theSmoother=      new KFTrajectorySmoother(*thePropagator,
					     *theUpdator,	
					     *theEstimator); 
  theMeasurementTracker = new MeasurementTracker(es);
  theLayerMeasurements  = new LayerMeasurements(theMeasurementTracker);

}

void CosmicTrajectoryBuilder::run(const TrajectorySeedCollection &collseed,
				  const SiStripRecHit2DLocalPosCollection &collstereo,
				  const SiStripRecHit2DLocalPosCollection &collrphi ,
				  const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const edm::EventSetup& es,
				  edm::Event& e,
				  vector<AlgoProduct> &algooutput)
{


  theMeasurementTracker->update(e);

  hits.clear();
  trajFit.clear();
  Acc_Z=0;
  Acc_Z2=0;
  //order all the hits
  vector<const TrackingRecHit*> allHits= SortHits(collstereo,collrphi,collmatched,collpixel,collseed);
  
  
  std::vector<Trajectory> trajSmooth;
  std::vector<Trajectory>::iterator trajIter;

  
  TrajectorySeedCollection::const_iterator iseed;
  for(iseed=collseed.begin();iseed!=collseed.end();iseed++){
 
    Trajectory startingTraj = createStartingTrajectory(*iseed);
    AddHit(startingTraj,allHits);
  

    for (trajIter=trajFit.begin(); trajIter!=trajFit.end();trajIter++){
      trajSmooth=theSmoother->trajectories((*trajIter));
    }

    for (trajIter= trajSmooth.begin(); trajIter!=trajSmooth.end();trajIter++){
      AlgoProduct tk=makeTrack((*trajIter));
      algooutput.push_back(tk);
   
    }
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
				  const SiPixelRecHitCollection &collpixel,
				  const TrajectorySeedCollection &collseed){

  vector<const TrackingRecHit*> allHits;
  vector<const TrackingRecHit*> seedHits;
  SiPixelRecHitCollection::const_iterator ipix;
  for(ipix=collpixel.begin();ipix!=collpixel.end();ipix++){
    allHits.push_back(&(*ipix));
  }

 
  SiStripRecHit2DLocalPosCollection::const_iterator istrip;
  TrajectorySeedCollection::const_iterator iseed;
  TrajectorySeedCollection::const_iterator seedbegin=collseed.begin();
  TrajectorySeedCollection::const_iterator seedend=collseed.end();
  TrajectorySeed::const_iterator ihit;
  for(istrip=collrphi.begin();istrip!=collrphi.end();istrip++){
    bool differenthit= true;
    for (iseed=seedbegin;iseed!=seedend;iseed++){
      TrajectorySeed::range hitRange= (*iseed).recHits();
      for (ihit = hitRange.first; 
	   ihit != hitRange.second; ihit++) {
	if((*ihit).geographicalId()==(*istrip).geographicalId()) {
	  if(((*ihit).localPosition()-(*istrip).localPosition()).mag()<0.1)  differenthit=false;
	}
      }
    }
    if (differenthit) allHits.push_back(&(*istrip)); 
    else  seedHits.push_back(&(*istrip)); 
  }
  
  hits.push_back((RHBuilder->build(seedHits.back()))); 
  hits.push_back((RHBuilder->build(seedHits.front()))); 
  float zz1=(RHBuilder->build(seedHits.back()))->globalPosition().z();
  Acc_Z+=zz1;
  Acc_Z2+=zz1*zz1;
  float zz2=(RHBuilder->build(seedHits.front()))->globalPosition().z();
  Acc_Z+=zz2;
  Acc_Z2+=zz2*zz2;
  nhits=2;
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

void CosmicTrajectoryBuilder::AddHit(Trajectory &traj,
				     vector<const TrackingRecHit*>Hits){
				     //				edm::OwnVector<TransientTrackingRecHit> hits){
  
  TSOS pp( (&traj)->lastMeasurement().updatedState());
  //  Trajectory cachetraj=traj;
  TSOS currentState( traj.lastMeasurement().updatedState());
  bl=track->barrelLayers();
  
  //  unsigned int icosmhit=0;

  vector<TM> meas;
  //  while ( icosmhit<Hits.size()) {
  for (unsigned int icosmhit=0;icosmhit<Hits.size();icosmhit++){
    DetId detid= Hits[icosmhit]->geographicalId();
    unsigned int subid=detid.subdetId();

    if    (subid==  PixelSubdetector::PixelBarrel) indexlayer=PXBDetId(detid).layer()-1;     
    
    if    (subid==  StripSubdetector::TIB)  indexlayer=TIBDetId(detid).layer()+2;
    
    if    (subid== StripSubdetector::TOB)  indexlayer=TOBDetId(detid).layer()+6;

    meas=theLayerMeasurements->measurements(*(bl[indexlayer]),currentState, 
					    *thePropagator, 
					    *theEstimator);

    if (meas.size()>0){
      
      //    for( vector<TM>::const_iterator itm = meas.begin(); 
      //	 itm != meas.end(); itm++) {
      int hitsbef=traj.foundHits();
      TransientTrackingRecHit* tmphit=RHBuilder->build(Hits[icosmhit]);


      float tmpz=tmphit->globalPosition().z();
      float medz=Acc_Z/nhits;
      float devz=sqrt((Acc_Z2/nhits)-(medz*medz));
      if (abs(tmpz-medz)<(15+(5*devz))){


	Acc_Z= Acc_Z+tmpz;
	Acc_Z2+=tmpz*tmpz;
	nhits++;


	updateTrajectory( traj, *(meas.begin()),*tmphit);
	int hitsaft=traj.foundHits();
	if (hitsaft>hitsbef){
	  hits.push_back(&(*tmphit));
	}
      }

    }
    
    //    icosmhit++;
  }
  
  

  if ( qualityFilter( traj)){
    const TrajectorySeed& tmpseed=traj.seed();
    TSOS startingState=startingTSOS(tmpseed);
    trajFit = theFitter->fit(tmpseed,hits, startingState );

  }

}


void CosmicTrajectoryBuilder::updateTrajectory( Trajectory& traj,
						const TM& tm,
						const TransientTrackingRecHit& hit) const
{
  TSOS predictedState = tm.predictedState();

  TSOS prSt= thePropagator->propagate(predictedState,
				      hit.det()->surface());
  
  
  if (prSt.isValid()){
    traj.push( TM( predictedState, theUpdator->update( predictedState, hit),
		   &hit, tm.estimate()));  
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

std::pair<Trajectory, reco::Track>  CosmicTrajectoryBuilder::makeTrack(const Trajectory &traj){
  //MP must be checked
  TSOS innertsos = traj.lastMeasurement().updatedState();
  int charge = innertsos.charge();
  //MP
  int ndof =5;
  const GlobalTrajectoryParameters& gp = innertsos.globalParameters();
  GlobalPoint v = gp.position();
  GlobalVector p = gp.momentum();
  const CartesianTrajectoryError& cte = innertsos.cartesianError();
  AlgebraicSymMatrix m = cte.matrix();
  math::Error<6>::type cov;
  for( int i = 0; i < 6; ++i )
    for( int j = 0; j <= i; ++j )
      cov( i, j ) = m.fast( i + 1 , j + 1 );
  math::XYZVector mom( p.x(), p.y(), p.z() );
  math::XYZPoint  vtx( v.x(), v.y(), v.z() );   
  edm::LogInfo("RecoTracker/TrackProducer") << " RESULT Momentum "<< p<<"\n";
  edm::LogInfo("RecoTracker/TrackProducer") << " RESULT Vertex "<< v<<"\n";
  //    traj.foundHits()<<" "<<
  //    charge<<" "<<endl;
  //build the Track(chiSquared, ndof, found, invalid, lost, q, vertex, momentum, covariance)
 
  reco::Track theTrack (traj.chiSquared(), 
			int(ndof),//FIXME fix weight() in TrackingRecHit 
			traj.foundHits(),//FIXME to be fixed in Trajectory.h
			0, //FIXME no corresponding method in trajectory.h
			traj.lostHits(),//FIXME to be fixed in Trajectory.h
			charge, 
			vtx,
			mom,
			cov);

  AlgoProduct aProduct(traj,theTrack);
  return aProduct;
  //  return theTrack; 
}

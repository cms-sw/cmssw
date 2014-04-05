//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia
#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
using namespace std;
CosmicTrajectoryBuilder::CosmicTrajectoryBuilder(const edm::ParameterSet& conf) : conf_(conf) { 
  //minimum number of hits per tracks

  theMinHits=conf_.getParameter<int>("MinHits");
  //cut on chi2
  chi2cut=conf_.getParameter<double>("Chi2Cut");
  edm::LogInfo("CosmicTrackFinder")<<"Minimum number of hits "<<theMinHits<<" Cut on Chi2= "<<chi2cut;

  geometry=conf_.getUntrackedParameter<std::string>("GeometricStructure","STANDARD");
}


CosmicTrajectoryBuilder::~CosmicTrajectoryBuilder() {
}


void CosmicTrajectoryBuilder::init(const edm::EventSetup& es, bool seedplus){

  // FIXME: this is a memory leak generator

  //services
  es.get<IdealMagneticFieldRecord>().get(magfield);
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
 
  
  if (seedplus) { 	 
    seed_plus=true; 	 
    thePropagator=      new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) ); 	 
    thePropagatorOp=    new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );} 	 
  else {
    seed_plus=false;
    thePropagator=      new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) ); 	
    thePropagatorOp=    new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
  }
  
  theUpdator=       new KFUpdator();
  theEstimator=     new Chi2MeasurementEstimator(chi2cut);
  

  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);
  

  RHBuilder=   theBuilder.product();
  hitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(RHBuilder)->cloner();



  theFitter=        new KFTrajectoryFitter(*thePropagator,
					   *theUpdator,	
					   *theEstimator) ;
  theFitter->setHitCloner(&hitCloner);

  theSmoother=      new KFTrajectorySmoother(*thePropagatorOp,
					     *theUpdator,	
					     *theEstimator);
  theSmoother->setHitCloner(&hitCloner);
}

void CosmicTrajectoryBuilder::run(const TrajectorySeedCollection &collseed,
				  const SiStripRecHit2DCollection &collstereo,
				  const SiStripRecHit2DCollection &collrphi ,
				  const SiStripMatchedRecHit2DCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const edm::EventSetup& es,
				  edm::Event& e,
				  vector<Trajectory> &trajoutput)
{

 


  std::vector<Trajectory> trajSmooth;
  std::vector<Trajectory>::iterator trajIter;
  
  TrajectorySeedCollection::const_iterator iseed;
  unsigned int IS=0;
  for(iseed=collseed.begin();iseed!=collseed.end();iseed++){
    bool seedplus=((*iseed).direction()==alongMomentum);
    init(es,seedplus);
    hits.clear();
    trajFit.clear();
    trajSmooth.clear();
    //order all the hits
    vector<const TrackingRecHit*> allHits= SortHits(collstereo,collrphi,collmatched,collpixel,*iseed);
    Trajectory startingTraj = createStartingTrajectory(*iseed);
    AddHit(startingTraj,allHits);
    for (trajIter=trajFit.begin(); trajIter!=trajFit.end();trajIter++){
      trajSmooth=theSmoother->trajectories((*trajIter));
    }
    for (trajIter= trajSmooth.begin(); trajIter!=trajSmooth.end();trajIter++){
      if((*trajIter).isValid()){
	trajoutput.push_back((*trajIter));
      }
    }
    delete theUpdator;
    delete theEstimator;
    delete thePropagator;
    delete thePropagatorOp;
    delete theFitter;
    delete theSmoother;
    //Only the first 30 seeds are considered
    if (IS>30) return;
    IS++;

  }
}

Trajectory CosmicTrajectoryBuilder::createStartingTrajectory( const TrajectorySeed& seed) const
{
  Trajectory result( seed, seed.direction()); 
  std::vector<TM> && seedMeas = seedMeasurements(seed);
  for (auto & i : seedMeas) result.push(std::move(i));
  return result;
}


std::vector<TrajectoryMeasurement> 
CosmicTrajectoryBuilder::seedMeasurements(const TrajectorySeed& seed) const
{
  std::vector<TrajectoryMeasurement> result;
  TrajectorySeed::range hitRange = seed.recHits();
  for (TrajectorySeed::const_iterator ihit = hitRange.first; 
       ihit != hitRange.second; ihit++) {
    //RC TransientTrackingRecHit* recHit = RHBuilder->build(&(*ihit));
    TransientTrackingRecHit::RecHitPointer recHit = RHBuilder->build(&(*ihit));
    const GeomDet* hitGeomDet = (&(*tracker))->idToDet( ihit->geographicalId());
    TSOS invalidState(new BasicSingleTrajectoryState( hitGeomDet->surface()));

    if (ihit == hitRange.second - 1) {
      TSOS  updatedState=startingTSOS(seed);
      result.emplace_back(invalidState, updatedState, recHit);
    } 
    else {
      result.emplace_back(invalidState, recHit);
    }
    
  }

  return result;
}





vector<const TrackingRecHit*> 
CosmicTrajectoryBuilder::SortHits(const SiStripRecHit2DCollection &collstereo,
				  const SiStripRecHit2DCollection &collrphi ,
				  const SiStripMatchedRecHit2DCollection &collmatched,
				  const SiPixelRecHitCollection &collpixel,
				  const TrajectorySeed &seed){


  //The Hits with global y more than the seed are discarded
  //The Hits correspondign to the seed are discarded
  //At the end all the hits are sorted in y
  vector<const TrackingRecHit*> allHits;

  SiStripRecHit2DCollection::DataContainer::const_iterator istrip;
  TrajectorySeed::range hRange= seed.recHits();
  TrajectorySeed::const_iterator ihit;
  float yref=0.;
  for (ihit = hRange.first; 
       ihit != hRange.second; ihit++) {
    yref=RHBuilder->build(&(*ihit))->globalPosition().y();
    hits.push_back((RHBuilder->build(&(*ihit)))); 
    LogDebug("CosmicTrackFinder")<<"SEED HITS"<<RHBuilder->build(&(*ihit))->globalPosition();
  }

  
  if ((&collpixel)!=0){
    SiPixelRecHitCollection::DataContainer::const_iterator ipix;
    for(ipix=collpixel.data().begin();ipix!=collpixel.data().end();ipix++){
      float ych= RHBuilder->build(&(*ipix))->globalPosition().y();
      if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	allHits.push_back(&(*ipix));
    }
  } 
  
  

  if ((&collrphi)!=0){
    for(istrip=collrphi.data().begin();istrip!=collrphi.data().end();istrip++){
      float ych= RHBuilder->build(&(*istrip))->globalPosition().y();
      if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	allHits.push_back(&(*istrip));   
    }
  }




  if ((&collstereo)!=0){
    for(istrip=collstereo.data().begin();istrip!=collstereo.data().end();istrip++){
      float ych= RHBuilder->build(&(*istrip))->globalPosition().y();
      if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
	allHits.push_back(&(*istrip));
    }
  }

//   SiStripMatchedRecHit2DCollection::DataContainer::const_iterator istripm;
//   if ((&collmatched)!=0){
//     for(istripm=collmatched.data().begin();istripm!=collmatched.data().end();istripm++){
//       float ych= RHBuilder->build(&(*istripm))->globalPosition().y();
//       if ((seed_plus && (ych<yref)) || (!(seed_plus) && (ych>yref)))
// 	allHits.push_back(&(*istripm));
//     }
//   }

  if (seed_plus){
    stable_sort(allHits.begin(),allHits.end(),CompareHitY_plus(*tracker));
  }
  else {
    stable_sort(allHits.begin(),allHits.end(),CompareHitY(*tracker));
  }

  return allHits;
}

TrajectoryStateOnSurface
CosmicTrajectoryBuilder::startingTSOS(const TrajectorySeed& seed)const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = (&(*tracker))->idToDet(DetId(pState.detId()));
  TSOS  State= trajectoryStateTransform::transientState( pState, &(gdet->surface()), 
					   &(*magfield));
  return State;

}

void CosmicTrajectoryBuilder::AddHit(Trajectory &traj,
				     const vector<const TrackingRecHit*>&Hits){


  unsigned int icosm2;
  unsigned int ibestdet;
  float chi2min;
  for (unsigned int icosmhit=0;icosmhit<Hits.size();icosmhit++){
    GlobalPoint gphit=RHBuilder->build(Hits[icosmhit])->globalPosition();
    unsigned int iraw= Hits[icosmhit]->geographicalId().rawId();
    LogDebug("CosmicTrackFinder")<<" HIT POSITION "<< gphit;
    //RC TransientTrackingRecHit* tmphit=RHBuilder->build(Hits[icosmhit]);
    TransientTrackingRecHit::RecHitPointer tmphit=RHBuilder->build(Hits[icosmhit]);
     TSOS prSt= thePropagator->propagate(traj.lastMeasurement().updatedState(),
 					tracker->idToDet(Hits[icosmhit]->geographicalId())->surface());

    //After propagating the trajectory to a detector,
    //find the most compatible hit in the det
    chi2min=20000000;
    ibestdet=1000;
     if (prSt.isValid()){
       LocalPoint  prLoc = prSt.localPosition();
       LogDebug("CosmicTrackFinder") <<"STATE PROPAGATED AT DET "<<iraw<<" "<<prSt;
       for(icosm2=icosmhit;icosm2<Hits.size();icosm2++){

	 if (iraw==Hits[icosm2]->geographicalId().rawId()){	  
	   TransientTrackingRecHit::RecHitPointer tmphit=RHBuilder->build(Hits[icosm2]);
	   float contr= theEstimator->estimate(prSt, *tmphit).second;
	   if (contr<chi2min) {
	     chi2min=contr;
	     ibestdet=icosm2;
	   }
	   if (icosm2!=icosmhit)	icosmhit++;

	 }
	 else  icosm2=Hits.size();
       }

       if(chi2min<chi2cut) 
	 LogDebug("CosmicTrackFinder")<<"Chi2 contribution for hit at "
				      <<RHBuilder->build(Hits[ibestdet])->globalPosition()
				      <<" is "<<chi2min;

       if(traj.foundHits()<3 &&(chi2min<chi2cut)){
	 //check on the first hit after the seed
 	 GlobalVector ck=RHBuilder->build(Hits[ibestdet])->globalPosition()-
	   traj.firstMeasurement().updatedState().globalPosition();
	 if ((abs(ck.x()/ck.y())>2)||(abs(ck.z()/ck.y())>2))  chi2min=300;
       }
       if (chi2min<chi2cut){
	 if ( abs(prLoc.x()) < 25  && abs(prLoc.y()) < 25 ){
	   TransientTrackingRecHit::RecHitPointer tmphitbestdet=RHBuilder->build(Hits[ibestdet]);
	   TSOS UpdatedState= theUpdator->update( prSt, *tmphitbestdet);
	   if (UpdatedState.isValid()){

	     traj.push(std::move(TM(prSt,UpdatedState,RHBuilder->build(Hits[ibestdet])
			  , chi2min)));
	     LogDebug("CosmicTrackFinder") <<
	       "STATE UPDATED WITH HIT AT POSITION "
					   <<tmphitbestdet->globalPosition()
					   <<UpdatedState<<" "
					   <<traj.chiSquared();

	     hits.push_back(tmphitbestdet);
	   }
	 }else LogDebug("CosmicTrackFinder")<<" Hits outside module surface "<< prLoc;
       }else LogDebug("CosmicTrackFinder")<<" State can not be updated with hit at position " <<gphit;
     }else LogDebug("CosmicTrackFinder")<<" State can not be propagated at det "<< iraw;    
  }
  
  

  if ( qualityFilter( traj)){
    const TrajectorySeed& tmpseed=traj.seed();
    if (thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
				   tracker->idToDet((*hits.begin())->geographicalId())->surface()).isValid()){
      TSOS startingState= 
	thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
				   tracker->idToDet((*hits.begin())->geographicalId())->surface());
      
      trajFit = theFitter->fit(tmpseed,hits, startingState );
    }
  }
  
}


bool 
CosmicTrajectoryBuilder::qualityFilter(const Trajectory& traj){
  int ngoodhits=0;
  if(geometry=="MTCC"){
    auto hits = traj.recHits();
    for(auto hit=hits.begin();hit!=hits.end();hit++){
      unsigned int iid=(*hit)->hit()->geographicalId().rawId();
      //CHECK FOR 3 hits r-phi
      if(((iid>>0)&0x3)!=1) ngoodhits++;
    }
  }
  else ngoodhits=traj.foundHits();
  
  if ( ngoodhits >= theMinHits) {
    return true;
  }
  else {
    return false;
  }
}

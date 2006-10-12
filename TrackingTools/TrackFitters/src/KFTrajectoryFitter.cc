#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
  
KFTrajectoryFitter::~KFTrajectoryFitter() {

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}


std::vector<Trajectory> KFTrajectoryFitter::fit(const Trajectory& aTraj) const {

  if(aTraj.empty()) return std::vector<Trajectory>();
 
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fit(aTraj.seed(), aTraj.recHits(), firstTsos);
}

std::vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
					   const RecHitContainer& hits) const{

  throw cms::Exception("TrackingTools/TrackFitters", 
		       "KFTrajectoryFitter::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return std::vector<Trajectory>();
}

std::vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
					   const RecHitContainer& hits,
					   const TSOS& firstPredTsos) const {


  if(hits.empty()) return std::vector<Trajectory>();
  LogDebug("TrackingTools/TrackFitters")
    <<" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    <<" KFTrajectoryFitter::fit staring with "<<hits.size()<<" HITS \n"
    <<" INITIAL STATE "<<firstPredTsos<<"\n";
  
  Trajectory myTraj(aSeed, propagator()->propagationDirection());


  TSOS predTsos(firstPredTsos);
  if(!predTsos.isValid()) {
    edm::LogError("TrackingTools/TrackFitters") 
      << "KFTrajectoryFitter: predicted tsos of first measurement not valid!\n"
      << "predTsos" << predTsos << "\n";
    return std::vector<Trajectory>();
  } 
  TSOS currTsos;
  if (hits.front()->isValid()) {
    //update
    TransientTrackingRecHit::RecHitPointer preciseHit = hits.front()->clone(predTsos);
    currTsos = updator()->update(predTsos, *preciseHit);
    myTraj.push(TM(predTsos, currTsos, preciseHit,
		   estimator()->estimate(predTsos, *preciseHit).second));
  } else {
    currTsos = predTsos;
    myTraj.push(TM(predTsos, *hits.begin() ));
  }
  const TransientTrackingRecHit & firsthit = *hits.front();
  
  if (firsthit.isValid()){
    
    LogDebug("TrackingTools/TrackFitters")
      <<" ----------------- FIRST HIT -----------------------\n"
      <<"  HIT IS AT R   "<<(firsthit).globalPosition().perp()<<"\n"
      <<"  HIT IS AT Z   "<<(firsthit).globalPosition().z()<<"\n"
      <<"  HIT IS AT Phi "<<(firsthit).globalPosition().phi()<<"\n"
      <<"  HIT IS AT Loc "<<(firsthit).localPosition()<<"\n"
      <<"  WITH LocError "<<(firsthit).localPositionError()<<"\n"
      <<"  HIT IS AT Glo "<<(firsthit).globalPosition()<<"\n"
      <<"SURFACE POSITION"<<"\n"
      <<(firsthit).det()->surface().position()<<"\n"
      <<"SURFACE ROTATION"<<"\n"
      <<(firsthit).det()->surface().rotation()<<"\n"
      <<" predTsos !"<<"\n"
      <<predTsos<<"\n"
      <<" currTsos !"<<"\n"
      <<currTsos<<"\n";
    LogDebug("TrackingTools/TrackFitters") <<"  GOING TO examine hit "<<(firsthit).geographicalId().rawId()<<"\n";
    if ((firsthit).geographicalId().subdetId() == StripSubdetector::TIB ) {
      LogDebug("TrackingTools/TrackFitters") <<" I am TIB "<<TIBDetId((firsthit).geographicalId()).layer()<<"\n";
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TOB ) { 
      LogDebug("TrackingTools/TrackFitters") <<" I am TOB "<<TOBDetId((firsthit).geographicalId()).layer()<<"\n";
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TEC ) { 
      LogDebug("TrackingTools/TrackFitters") <<" I am TEC "<<TECDetId((firsthit).geographicalId()).wheel()<<"\n";
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TID ) { 
      LogDebug("TrackingTools/TrackFitters") <<" I am TID "<<TIDDetId((firsthit).geographicalId()).wheel()<<"\n";
    }else{
      LogDebug("TrackingTools/TrackFitters") <<" I am Pixel "<<"\n";
    }
  }

  for(RecHitContainer::const_iterator ihit = hits.begin() + 1; 
      ihit != hits.end(); ihit++) {
    if ((**ihit).isValid() == false && (**ihit).det() == 0) {
      LogDebug("TrackingTools/TrackFitters") << " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }

    if ((**ihit).isValid()){
      LogDebug("TrackingTools/TrackFitters")
	<<" ----------------- NEW HIT -----------------------"<<"\n"
	<<"  HIT IS AT R   "<<(**ihit).globalPosition().perp()<<"\n"
	<<"  HIT IS AT Z   "<<(**ihit).globalPosition().z()<<"\n"
	<<"  HIT IS AT Phi "<<(**ihit).globalPosition().phi()<<"\n"
	<<"  HIT IS AT Loc "<<(**ihit).localPosition()<<"\n"
	<<"  WITH LocError "<<(**ihit).localPositionError()<<"\n"
	<<"  HIT IS AT Glo "<<(**ihit).globalPosition()<<"\n"
	<<"SURFACE POSITION"<<"\n"
	<<(**ihit).det()->surface().position()<<"\n"
	<<"SURFACE ROTATION"<<"\n"
	<<(**ihit).det()->surface().rotation()<<"\n";
      LogDebug("TrackingTools/TrackFitters") <<" GOING TO examine hit "<<(**ihit).geographicalId().rawId()<<"\n";
      if ((**ihit).geographicalId().subdetId() == StripSubdetector::TIB ) {
	LogDebug("TrackingTools/TrackFitters") <<" I am TIB "<<TIBDetId((**ihit).geographicalId()).layer()<<"\n";
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TOB ) { 
	LogDebug("TrackingTools/TrackFitters") <<" I am TOB "<<TOBDetId((**ihit).geographicalId()).layer()<<"\n";
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TEC ) { 
	LogDebug("TrackingTools/TrackFitters") <<" I am TEC "<<TECDetId((**ihit).geographicalId()).wheel()<<"\n";
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TID ) { 
	LogDebug("TrackingTools/TrackFitters") <<" I am TID "<<TIDDetId((**ihit).geographicalId()).wheel()<<"\n";
      }else{
	LogDebug("TrackingTools/TrackFitters") <<" I am Pixel "<<"\n";
      }
    }

    predTsos = propagator()->propagate(currTsos,
				       (**ihit).det()->surface());
    if(!predTsos.isValid()) {
      edm::LogError("TrackingTools/TrackFitters") 
	<<" SOMETHING WRONG !"<<"\n"
	<<"KFTrajectoryFitter: predicted tsos not valid!\n" 
	<<"current TSOS: "<<currTsos<< "\n";
      if((**ihit).isValid())
	edm::LogError("TrackingTools/TrackFitters")
	  << "next Surface: "<<(**ihit).det()->surface().position()<< "\n";
      return std::vector<Trajectory>();
    }
    if((**ihit).isValid()) {
      //update
      LogDebug("TrackingTools/TrackFitters") <<"THE HIT IS VALID: updating predTsos"<<"\n";
      TransientTrackingRecHit::RecHitPointer preciseHit = (**ihit).clone(predTsos);

      if (preciseHit->isValid() == false){
	LogDebug("TrackingTools/TrackFitters") <<"THE Precise HIT IS NOT VALID: using currTsos"<<"\n";
	currTsos = predTsos;
	myTraj.push(TM(predTsos, *ihit ));
      }else{
	
	currTsos = updator()->update(predTsos, *preciseHit);
	myTraj.push(TM(predTsos, currTsos, preciseHit,
		       estimator()->estimate(predTsos, *preciseHit).second));
      }
    } else {
      LogDebug("TrackingTools/TrackFitters") <<"THE HIT IS NOT VALID: using currTsos"<<"\n";
      currTsos = predTsos;
      myTraj.push(TM(predTsos, *ihit));
    }
    LogDebug("TrackingTools/TrackFitters")
      <<" predTsos !"<<"\n"
      <<predTsos<<"\n"
      <<" currTsos !"<<"\n"
      <<currTsos<<"\n";
    //std::cout <<(**ihit).det()->surface().position()<<std::endl;
  }
  //
  // debug
  //
  //std::cout <<" Before RETURN IN KFTrajectoryFitter"<<std::endl;
  
  LogDebug("TrackingTools/TrackFitters") <<" Found 1 trajectory wit hits "<< myTraj.foundHits()<<"\n";
  
  return std::vector<Trajectory>(1, myTraj);
}


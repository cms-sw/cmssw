#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

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

  throw cms::Exception("TrackFitters", 
		       "KFTrajectoryFitter::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return std::vector<Trajectory>();
}

std::vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
						const RecHitContainer& hits,
						const TSOS& firstPredTsos) const 
{
  if(hits.empty()) return std::vector<Trajectory>();

  if (aSeed.direction() == alongMomentum) {
    thePropagator->setPropagationDirection(alongMomentum);
  } else if (aSeed.direction() == oppositeToMomentum){
    thePropagator->setPropagationDirection(oppositeToMomentum);
  } else {
    throw cms::Exception("KFTrajectoryFitter","TrajectorySeed::direction() requested but not set");
  }



  LogDebug("TrackFitters")
    <<" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    <<" KFTrajectoryFitter::fit starting with "<<hits.size()<<" HITS \n"
    <<" INITIAL STATE "<<firstPredTsos;
  
  Trajectory myTraj(aSeed, thePropagator->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if(!predTsos.isValid()) {
    LogDebug("TrackFitters") 
      << "KFTrajectoryFitter: predicted tsos of first measurement not valid!\n"
      << "predTsos" << predTsos;
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
    LogDebug("TrackFitters")
      <<" ----------------- FIRST HIT -----------------------\n"
      <<"  HIT IS AT R   "<<(firsthit).globalPosition().perp()<<"\n"
      <<"  HIT IS AT Z   "<<(firsthit).globalPosition().z()<<"\n"
      <<"  HIT IS AT Phi "<<(firsthit).globalPosition().phi()<<"\n"
      <<"  HIT IS AT Loc "<<(firsthit).localPosition()<<"\n"
      <<"  WITH LocError "<<(firsthit).localPositionError()<<"\n"
      <<"  HIT IS AT Glo "<<(firsthit).globalPosition()<<"\n"
      <<"  HIT parameters "<<(firsthit).parameters()<<"\n"
      <<"  HIT parametersError "<<(firsthit).parametersError()<<"\n"
      <<"SURFACE POSITION"<<"\n"
      <<(firsthit).surface()->position()<<"\n"
      <<"SURFACE ROTATION"<<"\n"
      <<(firsthit).surface()->rotation();
    LogTrace("TrackFitters") <<" hit id="<<(firsthit).geographicalId().rawId();
    if ((firsthit).geographicalId().subdetId() == StripSubdetector::TIB ) {
      LogTrace("TrackFitters") <<" I am TIB "<<TIBDetId((firsthit).geographicalId()).layer();
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TOB ) { 
      LogTrace("TrackFitters") <<" I am TOB "<<TOBDetId((firsthit).geographicalId()).layer();
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TEC ) { 
      LogTrace("TrackFitters") <<" I am TEC "<<TECDetId((firsthit).geographicalId()).wheel();
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TID ) { 
      LogTrace("TrackFitters") <<" I am TID "<<TIDDetId((firsthit).geographicalId()).wheel();
    }else if ((firsthit).geographicalId().subdetId() == (int)PixelSubdetector::PixelBarrel ) {
      LogTrace("TrackFitters") <<" I am PixBar "<< PXBDetId((firsthit).geographicalId()).layer();
    }
    else {
      LogTrace("TrackFitters") <<" I am PixFwd "<< PXFDetId((firsthit).geographicalId()).disk();
    }
    LogTrace("TrackFitters")
      <<" predTsos !"<<"\n"
      <<predTsos<<"\n"
      <<" currTsos !"<<"\n"
      <<currTsos<<"\n";
  }

  for(RecHitContainer::const_iterator ihit = hits.begin() + 1; ihit != hits.end(); ihit++) {

    if ((**ihit).isValid() == false && (**ihit).surface() == 0) {
	LogDebug("TrackFitters")<< " Error: invalid hit with no GeomDet attached .... skipping";
	continue;
    }

    if ((**ihit).isValid()){
      LogDebug("TrackFitters")
	<<" ----------------- NEW HIT -----------------------"<<"\n"
	<<"  HIT IS AT R   "<<(**ihit).globalPosition().perp()<<"\n"
	<<"  HIT IS AT Z   "<<(**ihit).globalPosition().z()<<"\n"
	<<"  HIT IS AT Phi "<<(**ihit).globalPosition().phi()<<"\n"
	<<"  HIT IS AT Loc "<<(**ihit).localPosition()<<"\n"
	<<"  WITH LocError "<<(**ihit).localPositionError()<<"\n"
	<<"  HIT IS AT Glo "<<(**ihit).globalPosition()<<"\n"
	<<"SURFACE POSITION"<<"\n"
	<<(**ihit).surface()->position()<<"\n"
	<<"SURFACE ROTATION"<<"\n"
	<<(**ihit).surface()->rotation();
      LogTrace("TrackFitters") <<" hit det="<<(**ihit).geographicalId().rawId();
      if ((**ihit).geographicalId().subdetId() == StripSubdetector::TIB ) {
	LogTrace("TrackFitters") <<" I am TIB "<<TIBDetId((**ihit).geographicalId()).layer();
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TOB ) { 
	LogTrace("TrackFitters") <<" I am TOB "<<TOBDetId((**ihit).geographicalId()).layer();
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TEC ) { 
	LogTrace("TrackFitters") <<" I am TEC "<<TECDetId((**ihit).geographicalId()).wheel();
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TID ) { 
	LogTrace("TrackFitters") <<" I am TID "<<TIDDetId((**ihit).geographicalId()).wheel();
      }else if ((**ihit).geographicalId().subdetId() == StripSubdetector::TID ) { 
	LogTrace("TrackFitters") <<" I am TID "<<TIDDetId((**ihit).geographicalId()).wheel();
      }else if ((**ihit).geographicalId().subdetId() == (int) PixelSubdetector::PixelBarrel ) {
	LogTrace("TrackFitters") <<" I am PixBar "<< PXBDetId((**ihit).geographicalId()).layer();
      }
      else {
	LogTrace("TrackFitters") <<" I am PixFwd "<< PXFDetId((**ihit).geographicalId()).disk();
      }
    }

    predTsos = thePropagator->propagate( currTsos, *((**ihit).surface()) );

    if(!predTsos.isValid()) {
      LogDebug("TrackFitters") 
	<<" SOMETHING WRONG !"<<"\n"
	<<"KFTrajectoryFitter: predicted tsos not valid!\n" 
	<<"current TSOS: "<<currTsos<< "\n";
      if((**ihit).isValid())
	LogTrace("TrackFitters")
	  << "next Surface: "<<(**ihit).surface()->position()<< "\n";
      
      // this number could be made configurable
      if(myTraj.foundHits() >= 3)
	{
	  LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	  break;      
	}
      else      
	{        
	  LogDebug("TrackFitters") << " killing trajectory" << "\n";	   
	  return std::vector<Trajectory>();
	}
    }
    if((**ihit).isValid()) {
      //update
      LogTrace("TrackFitters") <<"THE HIT IS VALID: updating predTsos";
      TransientTrackingRecHit::RecHitPointer preciseHit = (**ihit).clone(predTsos);

      if (preciseHit->isValid() == false){
	LogDebug("TrackFitters") <<"THE Precise HIT IS NOT VALID: using currTsos"<<"\n";
	currTsos = predTsos;
	myTraj.push(TM(predTsos, *ihit ));
      }else{
	currTsos = updator()->update(predTsos, *preciseHit);
	myTraj.push(TM(predTsos, currTsos, preciseHit,
		       estimator()->estimate(predTsos, *preciseHit).second));
      }
    } else {
      LogDebug("TrackFitters") <<"THE HIT IS NOT VALID: using currTsos"<<"\n";
      currTsos = predTsos;
      myTraj.push(TM(predTsos, *ihit));
    }
    LogTrace("TrackFitters")
      <<" predTsos !"<<"\n"
      << predTsos <<"\n"
      <<" currTsos !"<<"\n"
      << currTsos;
  }
  
  LogDebug("TrackFitters") <<" Found 1 trajectory wit hits "<< myTraj.foundHits()<<"\n";
  
  return std::vector<Trajectory>(1, myTraj);
}


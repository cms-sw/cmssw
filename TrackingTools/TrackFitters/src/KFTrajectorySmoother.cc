#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

KFTrajectorySmoother::~KFTrajectorySmoother() {

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}

std::vector<Trajectory> 
KFTrajectorySmoother::trajectories(const Trajectory& aTraj) const {

  if(aTraj.empty()) return std::vector<Trajectory>();

  if (  aTraj.direction() == alongMomentum) {
    thePropagator->setPropagationDirection(oppositeToMomentum);
  }
  else {
    thePropagator->setPropagationDirection(alongMomentum);
  }



  Trajectory myTraj(aTraj.seed(), thePropagator->propagationDirection());

  std::vector<TM> avtm = aTraj.measurements();
  LogDebug("TrackFitters") <<" KFTrajectorySmoother::trajectories starting with "<< avtm.size() <<" HITS\n";

  for (unsigned int j=0;j<avtm.size();j++) { 
    if (avtm[j].recHit()->det()) 
      LogTrace("TrackFitters") << "hit #:" << j << " rawId=" << avtm[j].recHit()->det()->geographicalId().rawId() 
			       << " validity=" << avtm[j].recHit()->isValid();
    else
      LogTrace("TrackFitters") << "hit #:" << j << " Hit with no Det information";
  }

  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);

  if(!predTsos.isValid()) {
    LogDebug("TrackFitters") << 
      "KFTrajectorySmoother: predicted tsos of last measurement not valid!";
    return std::vector<Trajectory>();
  }
  TSOS currTsos;

  //first smoothed tm is last fitted
  if(avtm.back().recHit()->isValid()) {
    //update
    currTsos = updator()->update(predTsos, *(avtm.back().recHit()));
    const TransientTrackingRecHit & firsthit = *(avtm.back().recHit());
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
      <<(firsthit).surface().position()<<"\n"
      <<"SURFACE ROTATION"<<"\n"
      <<(firsthit).surface().rotation();
    LogTrace("TrackFitters") <<" hit id="<<(firsthit).geographicalId().rawId();
    if ((firsthit).geographicalId().subdetId() == StripSubdetector::TIB ) {
      LogTrace("TrackFitters") <<" I am TIB "<<TIBDetId((firsthit).geographicalId()).layer();
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TOB ) { 
      LogTrace("TrackFitters") <<" I am TOB "<<TOBDetId((firsthit).geographicalId()).layer();
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TEC ) { 
      LogTrace("TrackFitters") <<" I am TEC "<<TECDetId((firsthit).geographicalId()).wheel();
    }else if ((firsthit).geographicalId().subdetId() == StripSubdetector::TID ) { 
      LogTrace("TrackFitters") <<" I am TID "<<TIDDetId((firsthit).geographicalId()).wheel();
    }else if ((firsthit).geographicalId().subdetId() == (int) PixelSubdetector::PixelBarrel ) {
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
    

    myTraj.push(TM(avtm.back().forwardPredictedState(), 
		   predTsos,
		   avtm.back().updatedState(), 
		   avtm.back().recHit(),
		   avtm.back().estimate()), 
		avtm.back().estimate());
  } else {
    currTsos = predTsos;
    myTraj.push(TM(avtm.back().forwardPredictedState(),
		   avtm.back().recHit()));
  }
  
  TrajectoryStateCombiner combiner;

  for(std::vector<TM>::reverse_iterator itm = avtm.rbegin() + 1; 
      itm != avtm.rend() - 1; itm++) {
    LogDebug("TrackFitters")
      <<" ----------------- NEW HIT -----------------------"<<"\n"
      <<"  HIT IS AT R   "<<(*itm).recHit()->globalPosition().perp()<<"\n"
      <<"  HIT IS AT Z   "<<(*itm).recHit()->globalPosition().z()<<"\n"
      <<"  HIT IS AT Phi "<<(*itm).recHit()->globalPosition().phi()<<"\n"
      <<"  HIT IS AT Loc "<<(*itm).recHit()->localPosition()<<"\n"
      <<"  WITH LocError "<<(*itm).recHit()->localPositionError()<<"\n"
      <<"  HIT IS AT Glo "<<(*itm).recHit()->globalPosition()<<"\n"
      <<"SURFACE POSITION"<<"\n"
      <<(*itm).recHit()->surface().position()<<"\n"
      <<"SURFACE ROTATION"<<"\n"
      <<(*itm).recHit()->surface().rotation();
    LogTrace("TrackFitters") <<" hit id="<<(*itm).recHit()->geographicalId().rawId();
    if ((*itm).recHit()->geographicalId().subdetId() == StripSubdetector::TIB ) {
      LogTrace("TrackFitters") <<" I am TIB "<<TIBDetId((*itm).recHit()->geographicalId()).layer();
    }else if ((*itm).recHit()->geographicalId().subdetId() == StripSubdetector::TOB ) { 
      LogTrace("TrackFitters") <<" I am TOB "<<TOBDetId((*itm).recHit()->geographicalId()).layer();
    }else if ((*itm).recHit()->geographicalId().subdetId() == StripSubdetector::TEC ) { 
      LogTrace("TrackFitters") <<" I am TEC "<<TECDetId((*itm).recHit()->geographicalId()).wheel();
    }else if ((*itm).recHit()->geographicalId().subdetId() == StripSubdetector::TID ) { 
      LogTrace("TrackFitters") <<" I am TID "<<TIDDetId((*itm).recHit()->geographicalId()).wheel();
    }else if ((*itm).recHit()->geographicalId().subdetId() == (int) PixelSubdetector::PixelBarrel ) {
      LogTrace("TrackFitters") <<" I am PixBar "<< PXBDetId((*itm).recHit()->geographicalId()).layer();
    }
    else {
      LogTrace("TrackFitters") <<" I am PixFwd "<< PXFDetId((*itm).recHit()->geographicalId()).disk();
    }
    
    predTsos = thePropagator->propagate(currTsos,
					(*itm).recHit()->surface());
    
    if(!predTsos.isValid()) {
      LogDebug("TrackFitters") << 
	"KFTrajectorySmoother: predicted tsos not valid!";
      return std::vector<Trajectory>();
    }

    if((*itm).recHit()->isValid()) {

      //update
      currTsos = updator()->update(predTsos, (*(*itm).recHit()));
      //3 different possibilities to calculate smoothed state:
      //1: update combined predictions with hit
      //2: combine fwd-prediction with bwd-filter
      //3: combine bwd-prediction with fwd-filter
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      if(!combTsos.isValid()) {
	LogDebug("TrackFitters") << 
	  "KFTrajectorySmoother: combined tsos not valid!\n"<<
	  "pred Tsos pos: "<<predTsos.globalPosition()<< "\n" <<
	  "pred Tsos mom: "<<predTsos.globalMomentum()<< "\n" <<
	  "TrackingRecHit: "<<(*itm).recHit()->surface().toGlobal((*itm).recHit()->localPosition())<< "\n" ;
	return std::vector<Trajectory>();
      }

      TSOS smooTsos = combiner((*itm).updatedState(), predTsos);

      if(!smooTsos.isValid()) {
	LogDebug("TrackFitters") <<
	  "KFTrajectorySmoother: smoothed tsos not valid!";
	return std::vector<Trajectory>();
      }

      LogTrace("TrackFitters")
	<<" predTsos !"<<"\n"
	<< predTsos<<"\n"
	<<" currTsos !"<<"\n"
	<< currTsos<<"\n"
	<<" smooTsos !"<<"\n"
	<<  smooTsos<<"\n";
	
      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     smooTsos,
		     (*itm).recHit(),
		     estimator()->estimate(combTsos, *((*itm).recHit()) ).second),
		  (*itm).estimate());

    } else {
      currTsos = predTsos;
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      
      if(!combTsos.isValid()) {
    	LogDebug("TrackFitters") << 
    	  "KFTrajectorySmoother: combined tsos not valid!";
    	return std::vector<Trajectory>();
      }
      
      myTraj.push(TM((*itm).forwardPredictedState(),
    		     predTsos,
    		     combTsos,
    		     (*itm).recHit()));
    }
  }
  
  //last smoothed tm is last filtered
  predTsos = thePropagator->propagate(currTsos,
				      avtm.front().recHit()->surface());
  
  if(!predTsos.isValid()) {
	LogDebug("TrackFitters") << 
	  "KFTrajectorySmoother: predicted tsos not valid!";
    return std::vector<Trajectory>();
  }
  
  if(avtm.front().recHit()->isValid()) {

    //update
    currTsos = updator()->update(predTsos, *(avtm.front().recHit()));
    const TransientTrackingRecHit & lasthit = *(avtm.front().recHit());
    LogDebug("TrackFitters")
      <<" ----------------- LAST HIT -----------------------\n"
      <<"  HIT IS AT R   "<<(lasthit).globalPosition().perp()<<"\n"
      <<"  HIT IS AT Z   "<<(lasthit).globalPosition().z()<<"\n"
      <<"  HIT IS AT Phi "<<(lasthit).globalPosition().phi()<<"\n"
      <<"  HIT IS AT Loc "<<(lasthit).localPosition()<<"\n"
      <<"  WITH LocError "<<(lasthit).localPositionError()<<"\n"
      <<"  HIT IS AT Glo "<<(lasthit).globalPosition()<<"\n"
      <<"  HIT parameters "<<(lasthit).parameters()<<"\n"
      <<"  HIT parametersError "<<(lasthit).parametersError()<<"\n"
      <<"SURFACE POSITION"<<"\n"
      <<(lasthit).surface().position()<<"\n"
      <<"SURFACE ROTATION"<<"\n"
      <<(lasthit).surface().rotation();
    LogTrace("TrackFitters") <<"  GOING TO examine hit "<<(lasthit).geographicalId().rawId();
    if ((lasthit).geographicalId().subdetId() == StripSubdetector::TIB ) {
      LogTrace("TrackFitters") <<" I am TIB "<<TIBDetId((lasthit).geographicalId()).layer();
    }else if ((lasthit).geographicalId().subdetId() == StripSubdetector::TOB ) { 
      LogTrace("TrackFitters") <<" I am TOB "<<TOBDetId((lasthit).geographicalId()).layer();
    }else if ((lasthit).geographicalId().subdetId() == StripSubdetector::TEC ) { 
      LogTrace("TrackFitters") <<" I am TEC "<<TECDetId((lasthit).geographicalId()).wheel();
    }else if ((lasthit).geographicalId().subdetId() == StripSubdetector::TID ) { 
      LogTrace("TrackFitters") <<" I am TID "<<TIDDetId((lasthit).geographicalId()).wheel();
    }else if ((lasthit).geographicalId().subdetId() == (int) PixelSubdetector::PixelBarrel ) {
      LogTrace("TrackFitters") <<" I am PixBar "<< PXBDetId((lasthit).geographicalId()).layer();
    }
    else {
      LogTrace("TrackFitters") <<" I am PixFwd "<< PXFDetId((lasthit).geographicalId()).disk();
    }
    LogTrace("TrackFitters")
      <<" predTsos !"<<"\n"
      << predTsos<<"\n"
      <<" currTsos !"<<"\n"
      << currTsos<<"\n";

    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   estimator()->estimate(predTsos, *(avtm.front().recHit())).second),
		avtm.front().estimate());

  } else {
    myTraj.push(TM(avtm.front().forwardPredictedState(),
  		   avtm.front().recHit()));
  }
  
  return std::vector<Trajectory>(1, myTraj); 

}

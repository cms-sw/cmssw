#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


KFTrajectorySmoother::~KFTrajectorySmoother() {

  delete thePropagatorAlongMomentum;
  delete thePropagatorOppositeToMomentum;
  delete theUpdator;
  delete theEstimator;

}

std::vector<Trajectory> 
KFTrajectorySmoother::trajectories(const Trajectory& aTraj) const {

  if(aTraj.empty()) return std::vector<Trajectory>();

  const Propagator*  theBackwardPropagator;

  if (  aTraj.direction() == alongMomentum) {
    theBackwardPropagator = thePropagatorOppositeToMomentum;
  }
  else {
    theBackwardPropagator = thePropagatorAlongMomentum;
  }



  Trajectory myTraj(aTraj.seed(), theBackwardPropagator->propagationDirection());

  std::vector<TM> avtm = aTraj.measurements();

  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);

  if(!predTsos.isValid()) {
    LogDebug("TrackingTools/TrackFitters") << 
      "KFTrajectorySmoother: predicted tsos of last measurement not valid!";
    return std::vector<Trajectory>();
  }
  TSOS currTsos;

  //first smoothed tm is last fitted
  if(avtm.back().recHit()->isValid()) {
    currTsos = updator()->update(predTsos, *(avtm.back().recHit()));
    myTraj.push(TM(avtm.back().forwardPredictedState(), 
		   predTsos,
		   avtm.back().updatedState(), 
		   avtm.back().recHit(),
		   avtm.back().estimate()//,
		   /*avtm.back().layer()*/), 
		avtm.back().estimate());
  } else {
    currTsos = predTsos;
    myTraj.push(TM(avtm.back().forwardPredictedState(),
		   avtm.back().recHit()//,
		   /*avtm.back().layer()*/));
  }
  
  TrajectoryStateCombiner combiner;

  for(std::vector<TM>::reverse_iterator itm = avtm.rbegin() + 1; 
      itm != avtm.rend() - 1; itm++) {

    predTsos = theBackwardPropagator->propagate(currTsos,
				       (*itm).recHit()->det()->surface());

    if(!predTsos.isValid()) {
      LogDebug("TrackingTools/TrackFitters") << 
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
	LogDebug("TrackingTools/TrackFitters") << 
	  "KFTrajectorySmoother: combined tsos not valid!\n"<<
	  "pred Tsos pos: "<<predTsos.globalPosition()<< "\n" <<
	  "pred Tsos mom: "<<predTsos.globalMomentum()<< "\n" <<
	  "TrackingRecHit: "<<(*itm).recHit()->det()->surface().toGlobal((*itm).recHit()->localPosition())<< "\n" ;
	return std::vector<Trajectory>();
      }

      TSOS smooTsos = combiner((*itm).updatedState(), predTsos);

      if(!smooTsos.isValid()) {
	LogDebug("TrackingTools/TrackFitters") <<
	  "KFTrajectorySmoother: smoothed tsos not valid!";
	return std::vector<Trajectory>();
      }
      
      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     smooTsos,
		     (*itm).recHit(),
		     estimator()->estimate(combTsos, *((*itm).recHit()) ).second//,
		     /*(*itm).layer()*/),
		     (*itm).estimate());
      //estimator()->estimate(predTsos, (*itm).recHit()));
    } else {
      currTsos = predTsos;
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      
      if(!combTsos.isValid()) {
	LogDebug("TrackingTools/TrackFitters") << 
	  "KFTrajectorySmoother: combined tsos not valid!";
	return std::vector<Trajectory>();
      }

      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     combTsos,
		     (*itm).recHit()//,
		     /*(*itm).layer()*/));
    }
  }
  
  //last smoothed tm is last filtered
  predTsos = theBackwardPropagator->propagate(currTsos,
				     avtm.front().recHit()->det()->surface());
  
  if(!predTsos.isValid()) {
	LogDebug("TrackingTools/TrackFitters") << 
	  "KFTrajectorySmoother: predicted tsos not valid!";
    return std::vector<Trajectory>();
  }
  
  if(avtm.front().recHit()->isValid()) {
    //update
    currTsos = updator()->update(predTsos, *(avtm.front().recHit()));
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   estimator()->estimate(predTsos, *(avtm.front().recHit())).second//,
		   /*avtm.front().layer()*/),
		avtm.front().estimate());
    //estimator()->estimate(predTsos, avtm.front().recHit()));
  } else {
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   avtm.front().recHit()//,
		   /*avtm.front().layer()*/));
  }
  
  return std::vector<Trajectory>(1, myTraj); 

}

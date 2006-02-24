#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

KFTrajectoryFitter::~KFTrajectoryFitter() {

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}


vector<Trajectory> KFTrajectoryFitter::fit(const Trajectory& aTraj) const {

  if(aTraj.empty()) return vector<Trajectory>();
 
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fit(aTraj.seed(), aTraj.recHits(), firstTsos);
}

vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
					   const edm::OwnVector<TransientTrackingRecHit>& hits) const{

  throw cms::Exception("TrackingTools/TrackFitters", 
		       "KFTrajectoryFitter::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return vector<Trajectory>();
}

vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
					   const edm::OwnVector<TransientTrackingRecHit>& hits,
					   const TSOS& firstPredTsos) const {

  if(hits.empty()) return vector<Trajectory>();

  Trajectory myTraj(aSeed, propagator()->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if(!predTsos.isValid()) {
    LogDebug("TrackingTools/TrackFitters") << 
      "KFTrajectoryFitter: predicted tsos of first measurement not valid!";
    return vector<Trajectory>();
  } 
  TSOS currTsos;

  if((&*(hits.begin()))->isValid()) {
    //update
    currTsos = updator()->update(predTsos, *(hits.begin()));
    myTraj.push(TM(predTsos, currTsos, ((hits.begin())->clone() ),
		   estimator()->estimate(predTsos, *(hits.begin()) ).second));
  } else {
    currTsos = predTsos;
    myTraj.push(TM(predTsos, hits.begin()->clone() ));
  }
  
  for(edm::OwnVector<TransientTrackingRecHit>::const_iterator ihit = hits.begin() + 1; 
      ihit != hits.end(); ihit++) {

    predTsos = propagator()->propagate(currTsos,
				       (*ihit).detUnit()->surface());

    if(!predTsos.isValid()) {
      LogDebug("TrackingTools/TrackFitters") << 
	"KFTrajectoryFitter: predicted tsos not valid!\n" <<
	"current TSOS: "<<currTsos<< "\n";
      if((*ihit).isValid())
	LogDebug("TrackingTools/TrackFitters") << 
	  "next Surface: "<<(*ihit).detUnit()->surface().position()<< "\n";
      return vector<Trajectory>();
    }
    
    if((*ihit).isValid()) {
      //update
      currTsos = updator()->update(predTsos, *ihit);
      myTraj.push(TM(predTsos, currTsos, (*ihit).clone(),
		     estimator()->estimate(predTsos, *ihit).second));
    } else {
      currTsos = predTsos;
      myTraj.push(TM(predTsos, (*ihit).clone() ));
    }
  }
  
  return vector<Trajectory>(1, myTraj);
}





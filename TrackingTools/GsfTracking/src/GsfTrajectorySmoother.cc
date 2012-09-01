#include "TrackingTools/GsfTracking/interface/GsfTrajectorySmoother.h"

#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GsfTrajectorySmoother::GsfTrajectorySmoother(const GsfPropagatorWithMaterial& aPropagator,
					     const TrajectoryStateUpdator& aUpdator,
					     const MeasurementEstimator& aEstimator,
					     const MultiTrajectoryStateMerger& aMerger,
					     float errorRescaling,
					     const bool materialBeforeUpdate,
					     const DetLayerGeometry* detLayerGeometry) :
  thePropagator(aPropagator.clone()),
  theGeomPropagator(0),
  theConvolutor(0),
  theUpdator(aUpdator.clone()),
  theEstimator(aEstimator.clone()),
  theMerger(aMerger.clone()),
  theMatBeforeUpdate(materialBeforeUpdate),
  theErrorRescaling(errorRescaling),
  theGeometry(detLayerGeometry)
{
  if ( !theMatBeforeUpdate ) {
    theGeomPropagator = new GsfPropagatorAdapter(thePropagator->geometricalPropagator());
    theConvolutor = thePropagator->convolutionWithMaterial().clone();
  }

  if(!theGeometry) theGeometry = &dummyGeometry;

  //   static SimpleConfigurable<bool> timeConf(false,"GsfTrajectorySmoother:activateTiming");
  //   theTiming = timeConf.value();
}

GsfTrajectorySmoother::~GsfTrajectorySmoother() {
  delete thePropagator;
  delete theGeomPropagator;
  delete theConvolutor;
  delete theUpdator;
  delete theEstimator;
  delete theMerger;
}

Trajectory 
GsfTrajectorySmoother::trajectory(const Trajectory& aTraj) const {

  //   static TimingReport::Item* propTimer =
  //     &(*TimingReport::current())[string("GsfTrajectorySmoother:propagation")];
  //   propTimer->switchCPU(false);
  //   if ( !theTiming )  propTimer->switchOn(false);
  //   static TimingReport::Item* updateTimer =
  //     &(*TimingReport::current())[string("GsfTrajectorySmoother:update")];
  //   updateTimer->switchCPU(false);
  //   if ( !theTiming )  updateTimer->switchOn(false);
  
  if(aTraj.empty()) return Trajectory();
  
  if (  aTraj.direction() == alongMomentum) {
    thePropagator->setPropagationDirection(oppositeToMomentum);
  }
  else {
    thePropagator->setPropagationDirection(alongMomentum);
  }

  Trajectory myTraj(aTraj.seed(), propagator()->propagationDirection());
  
  std::vector<TM> const & avtm = aTraj.measurements();
  
  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);

  if(!predTsos.isValid()) {
    edm::LogInfo("GsfTrajectorySmoother") 
      << "GsfTrajectorySmoother: predicted tsos of last measurement not valid!";
    return Trajectory();
  }
  TSOS currTsos;

  //first smoothed tm is last fitted
  if(avtm.back().recHit()->isValid()) {
    {
    //       TimeMe t(*updateTimer,false);
    currTsos = updator()->update(predTsos, *avtm.back().recHit());
    }

    if(!currTsos.isValid()) {
      edm::LogInfo("GsfTrajectorySmoother") << "GsfTrajectorySmoother: tsos not valid after update!";
      return Trajectory();
    }
    
    //check validity
    if (!avtm.back().forwardPredictedState().isValid() || !predTsos.isValid() || !avtm.back().updatedState().isValid()){
      edm::LogError("InvalidState")<<"first hit";
      return Trajectory();
    }
    
    myTraj.push(TM(avtm.back().forwardPredictedState(), 
		   predTsos,
		   avtm.back().updatedState(),
		   avtm.back().recHit(),
		   avtm.back().estimate(),
		   theGeometry->idToLayer(avtm.back().recHit()->geographicalId()) ), 
		avtm.back().estimate());
  } else {
    currTsos = predTsos;
    //check validity
    if (!avtm.back().forwardPredictedState().isValid()){
      edm::LogError("InvalidState")<<"first hit on invalid hit";
      return Trajectory();
    }

    myTraj.push(TM(avtm.back().forwardPredictedState(),
		   avtm.back().recHit(),
		   0.,
		   theGeometry->idToLayer(avtm.back().recHit()->geographicalId() )  ));
  }
  
  TrajectoryStateCombiner combiner;

  for(std::vector<TM>::const_reverse_iterator itm = avtm.rbegin() + 1; 
      itm < avtm.rend() - 1; ++itm) {
    {
      //       TimeMe t(*propTimer,false);
      //       //
      //       // check type of surface in case of invalid hit
      //       // (in this version only propagations to planes are
      //       // supported for multi trajectory states)
      //       //
      //       if ( !(*itm).recHit().isValid() ) {
      // 	const BoundPlane* plane = 
      // 	  dynamic_cast<const BoundPlane*>(&(*itm).recHit().det().surface());
      // 	//
      // 	// no plane: insert invalid 
      // 	if ( plane==0 ) {
      // 	  myTraj.push(TM(TrajectoryStateOnSurface(),
      // 			 (*itm).recHit());
      // 	  continue;
      // 	}
      //     }
      predTsos = propagator()->propagate(currTsos,
					 *(*itm).recHit()->surface());
    }
    if ( predTsos.isValid() && theConvolutor && theMatBeforeUpdate )
      predTsos = (*theConvolutor)(predTsos,
				  propagator()->propagationDirection());
    if(!predTsos.isValid()) {
      edm::LogInfo("GsfTrajectorySmoother") << "GsfTrajectorySmoother: predicted tsos not valid!";
      return Trajectory();
    }
    if ( theMerger )  predTsos = theMerger->merge(predTsos);

    if((*itm).recHit()->isValid()) {
      //update
      {
	// 	TimeMe t(*updateTimer,false);
	currTsos = updator()->update(predTsos, *(*itm).recHit());
      }
      if ( currTsos.isValid() && theConvolutor && !theMatBeforeUpdate )
	currTsos = (*theConvolutor)(currTsos,
				    propagator()->propagationDirection());
      if(!currTsos.isValid()) {
	edm::LogInfo("GsfTrajectorySmoother") 
	  << "GsfTrajectorySmoother: tsos not valid after update / material effects!";
	return Trajectory();
      }
      //3 different possibilities to calculate smoothed state:
      //1: update combined predictions with hit
      //2: combine fwd-prediction with bwd-filter
      //3: combine bwd-prediction with fwd-filter
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      if(!combTsos.isValid()) {
	LogDebug("GsfTrajectorySmoother") << 
	  "KFTrajectorySmoother: combined tsos not valid!\n"<<
	  "pred Tsos pos: "<<predTsos.globalPosition()<< "\n" <<
	  "pred Tsos mom: "<<predTsos.globalMomentum()<< "\n" <<
	  "TrackingRecHit: "<<(*itm).recHit()->surface()->toGlobal((*itm).recHit()->localPosition())<< "\n" ;
	return Trajectory();
      }

      TSOS smooTsos = combiner((*itm).updatedState(), predTsos);

      if(!smooTsos.isValid()) {
	LogDebug("GsfTrajectorySmoother") <<
	  "KFTrajectorySmoother: smoothed tsos not valid!";
	return Trajectory();
      }

      if (!(*itm).forwardPredictedState().isValid() || !predTsos.isValid() || !smooTsos.isValid() ){
	edm::LogError("InvalidState")<<"inside hits with combination.";
	return Trajectory();
      }


      myTraj.push(TM((*itm).forwardPredictedState(),
		     predTsos,
		     smooTsos,
		     (*itm).recHit(),
		     estimator()->estimate(combTsos, *(*itm).recHit()).second,
		     theGeometry->idToLayer((*itm).recHit()->geographicalId() ) ),
		  (*itm).estimate());
    } 
    else {
      currTsos = predTsos;
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      
      if(!combTsos.isValid()) {
    	LogDebug("GsfTrajectorySmoother") << 
    	  "KFTrajectorySmoother: combined tsos not valid!";
    	return Trajectory();
      }

      if (!(*itm).forwardPredictedState().isValid() || !predTsos.isValid() || !combTsos.isValid() ){
	edm::LogError("InvalidState")<<"inside hits with invalid rechit.";
        return Trajectory();
      }

      myTraj.push(TM((*itm).forwardPredictedState(),
    		     predTsos,
    		     combTsos,
    		     (*itm).recHit(),
		     0.,
		     theGeometry->idToLayer((*itm).recHit()->geographicalId()) ));
    }
    if ( theMerger )  currTsos = theMerger->merge(currTsos);
  }

  //last smoothed tm is last filtered
  {
    //     TimeMe t(*propTimer,false);
    predTsos = propagator()->propagate(currTsos,
				       *avtm.front().recHit()->surface());
  }
  if ( predTsos.isValid() && theConvolutor && theMatBeforeUpdate )
    predTsos = (*theConvolutor)(predTsos,
				propagator()->propagationDirection());
  if(!predTsos.isValid()) {
    edm::LogInfo("GsfTrajectorySmoother") << "GsfTrajectorySmoother: predicted tsos not valid!";
    return Trajectory();
  }
  if ( theMerger )  predTsos = theMerger->merge(predTsos);

  if(avtm.front().recHit()->isValid()) {
    //update
    {
      //       TimeMe t(*updateTimer,false);
      currTsos = updator()->update(predTsos, *avtm.front().recHit());
    }
    if ( currTsos.isValid() && theConvolutor && !theMatBeforeUpdate )
      currTsos = (*theConvolutor)(currTsos,
				  propagator()->propagationDirection());
    if(!currTsos.isValid()) {
      edm::LogInfo("GsfTrajectorySmoother") 
	<< "GsfTrajectorySmoother: tsos not valid after update / material effects!";
      return Trajectory();
    }
  
    if (!avtm.front().forwardPredictedState().isValid() || !predTsos.isValid() || !currTsos.isValid() ){
      edm::LogError("InvalidState")<<"last hit";
      return Trajectory();
    }

    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   predTsos,
		   currTsos,
		   avtm.front().recHit(),
		   estimator()->estimate(predTsos, *avtm.front().recHit()).second,
		   theGeometry->idToLayer(avtm.front().recHit()->geographicalId() )),
		avtm.front().estimate());
    //estimator()->estimate(predTsos, avtm.front().recHit()));
  }
  else {
    if (!avtm.front().forwardPredictedState().isValid()){
      edm::LogError("InvalidState")<<"last invalid hit";
      return Trajectory();
    }
    myTraj.push(TM(avtm.front().forwardPredictedState(),
		   avtm.front().recHit(),
		   0.,
		   theGeometry->idToLayer(avtm.front().recHit()->geographicalId()) ));
  }

  return myTraj; 
}

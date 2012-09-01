#include "TrackingTools/GsfTracking/interface/GsfTrajectoryFitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
// #include "CommonDet/BasicDet/interface/Det.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
// #include "Utilities/Notification/interface/Verbose.h"
// #include "Utilities/Notification/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GsfTrajectoryFitter::GsfTrajectoryFitter(const Propagator& aPropagator,
					 const TrajectoryStateUpdator& aUpdator,
					 const MeasurementEstimator& aEstimator,
					 const MultiTrajectoryStateMerger& aMerger,
					 const DetLayerGeometry* detLayerGeometry) :
  thePropagator(aPropagator.clone()),
  theUpdator(aUpdator.clone()),
  theEstimator(aEstimator.clone()),
  theMerger(aMerger.clone()),
  theGeometry(detLayerGeometry)
{
  if(!theGeometry) theGeometry = &dummyGeometry;
  //   static SimpleConfigurable<bool> timeConf(false,"GsfTrajectoryFitter:activateTiming");
  //   theTiming = timeConf.value();
}

GsfTrajectoryFitter::~GsfTrajectoryFitter() {
  delete thePropagator;
  delete theUpdator;
  delete theEstimator;
  delete theMerger;
}

Trajectory GsfTrajectoryFitter::fitOne(const Trajectory& aTraj, fitType type) const {  
  if(aTraj.empty()) return Trajectory();
 
  TM const & firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fitOne(aTraj.seed(), aTraj.recHits(), firstTsos,type);
}

Trajectory GsfTrajectoryFitter::fitOne(const TrajectorySeed& aSeed,
				       const RecHitContainer& hits, fitType type) const {
  
  edm::LogError("GsfTrajectoryFitter") 
    << "GsfTrajectoryFitter::fit(TrajectorySeed, vector<RecHit>) not implemented";
  
  return Trajectory();
}

Trajectory GsfTrajectoryFitter::fitOne(const TrajectorySeed& aSeed,
				    const RecHitContainer& hits, 
				    const TrajectoryStateOnSurface& firstPredTsos,
				    fitType) const {

  //   static TimingReport::Item* propTimer =
  //     &(*TimingReport::current())[string("GsfTrajectoryFitter:propagation")];
  //   propTimer->switchCPU(false);
  //   if ( !theTiming )  propTimer->switchOn(false);
  //   static TimingReport::Item* updateTimer =
  //     &(*TimingReport::current())[string("GsfTrajectoryFitter:update")];
  //   updateTimer->switchCPU(false);
  //   if ( !theTiming )  updateTimer->switchOn(false);

  if(hits.empty()) return Trajectory();

  Trajectory myTraj(aSeed, propagator()->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if(!predTsos.isValid()) {
    edm::LogInfo("GsfTrajectoryFitter") 
      << "GsfTrajectoryFitter: predicted tsos of first measurement not valid!";
    return Trajectory();
  } 

  TSOS currTsos;
  if(hits.front()->isValid()) {
    //update
    TransientTrackingRecHit::RecHitPointer preciseHit = hits.front()->clone(predTsos);
    {
      //       TimeMe t(*updateTimer,false);
      currTsos = updator()->update(predTsos, *preciseHit);
    }
    if (!predTsos.isValid() || !currTsos.isValid()){
      edm::LogError("InvalidState")<<"first hit";
      return Trajectory();
    }
    myTraj.push(TM(predTsos, currTsos, preciseHit, 0., theGeometry->idToLayer(preciseHit->geographicalId() )),
		estimator()->estimate(predTsos, *preciseHit).second);
  } else {
    currTsos = predTsos;
    if (!predTsos.isValid()){
      edm::LogError("InvalidState")<<"first invalid hit";
      return Trajectory();
    }
    myTraj.push(TM(predTsos, *hits.begin(),0., theGeometry->idToLayer((*hits.begin())->geographicalId()) ));
  }
  
  for(RecHitContainer::const_iterator ihit = hits.begin() + 1; 
      ihit != hits.end(); ihit++) {    
    //
    // temporary protection copied from KFTrajectoryFitter.
    //
    if ((**ihit).isValid() == false && (**ihit).det() == 0) {
      LogDebug("GsfTrajectoryFitter") << " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }

    //!!! no invalid hits on cylinders anymore??
    //     //
    //     // check type of surface in case of invalid hit
    //     // (in this version only propagations to planes are
    //     // supported for multi trajectory states)
    //     //
    //     if ( !(**ihit).isValid() ) {
    //       const BoundPlane* plane = 
    // 	dynamic_cast<const BoundPlane*>(&(**ihit).det().surface());
    //       //
    //       // no plane: insert invalid measurement
    //       //
    //       if ( plane==0 ) {
    // 	myTraj.push(TM(TrajectoryStateOnSurface(),&(**ihit)));
    // 	continue;
    //       }
    //     }
    {
      //       TimeMe t(*propTimer,false);
      predTsos = propagator()->propagate(currTsos,
					 (**ihit).det()->surface());
    }
    if(!predTsos.isValid()) {
      if ( myTraj.foundHits()>=3 ) {
	edm::LogInfo("GsfTrajectoryFitter") 
	  << "GsfTrajectoryFitter: predicted tsos not valid! \n"
	  << "Returning trajectory with " << myTraj.foundHits() << " found hits.";
	return myTraj;
      }
      else {
      edm::LogInfo("GsfTrajectoryFitter") 
	<< "GsfTrajectoryFitter: predicted tsos not valid after " << myTraj.foundHits()
	<< " hits, discarding candidate!";
	return Trajectory();
      }
    }
    if ( merger() ) predTsos = merger()->merge(predTsos);
    
    if((**ihit).isValid()) {
      //update
      TransientTrackingRecHit::RecHitPointer preciseHit = (**ihit).clone(predTsos);
      {
	// 	TimeMe t(*updateTimer,false);
	currTsos = updator()->update(predTsos, *preciseHit);
      }
      if (!predTsos.isValid() || !currTsos.isValid()){
	edm::LogError("InvalidState")<<"inside hit";
	return Trajectory();
      }
      myTraj.push(TM(predTsos, currTsos, preciseHit,
		     estimator()->estimate(predTsos, *preciseHit).second,
		     theGeometry->idToLayer(preciseHit->geographicalId() )));
    } else {
      currTsos = predTsos;
      if (!predTsos.isValid()){
      edm::LogError("InvalidState")<<"inside invalid hit";
      return Trajectory();
      }
      myTraj.push(TM(predTsos, *ihit,0., theGeometry->idToLayer( (*ihit)->geographicalId()) ));
    }
  }
  return myTraj;
}

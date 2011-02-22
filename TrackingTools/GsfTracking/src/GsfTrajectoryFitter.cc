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

std::vector<Trajectory> GsfTrajectoryFitter::fit(const Trajectory& aTraj) const 
{  
  if(aTraj.empty()) return std::vector<Trajectory>();
 
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fit(aTraj.seed(), aTraj.recHits(), firstTsos);
}

std::vector<Trajectory> GsfTrajectoryFitter::fit(const TrajectorySeed& aSeed,
						 const RecHitContainer& hits) const {

  edm::LogError("GsfTrajectoryFitter") 
    << "GsfTrajectoryFitter::fit(TrajectorySeed, vector<RecHit>) not implemented";

  return std::vector<Trajectory>();
}

std::vector<Trajectory> GsfTrajectoryFitter::fit(const TrajectorySeed& aSeed,
						 const RecHitContainer& hits, 
						 const TSOS& firstPredTsos) const {

  //   static TimingReport::Item* propTimer =
  //     &(*TimingReport::current())[string("GsfTrajectoryFitter:propagation")];
  //   propTimer->switchCPU(false);
  //   if ( !theTiming )  propTimer->switchOn(false);
  //   static TimingReport::Item* updateTimer =
  //     &(*TimingReport::current())[string("GsfTrajectoryFitter:update")];
  //   updateTimer->switchCPU(false);
  //   if ( !theTiming )  updateTimer->switchOn(false);

  if(hits.empty()) return std::vector<Trajectory>();

  Trajectory myTraj(aSeed, propagator()->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if(!predTsos.isValid()) {
    edm::LogInfo("GsfTrajectoryFitter") 
      << "GsfTrajectoryFitter: predicted tsos of first measurement not valid!";
    return std::vector<Trajectory>();
  } 

  TSOS currTsos;
  if(hits.front()->isValid()) {
    //update
    TransientTrackingRecHit::RecHitPointer preciseHit = hits.front()->clone(predTsos);
    {
      //       TimeMe t(*updateTimer,false);
      currTsos = updator()->update(predTsos, *preciseHit);
    }
    myTraj.push(TM(predTsos, currTsos, preciseHit, 0., theGeometry->idToLayer(preciseHit->geographicalId() )),
		estimator()->estimate(predTsos, *preciseHit).second);
  } else {
    currTsos = predTsos;
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
	return std::vector<Trajectory>(1,myTraj);
      }
      else {
      edm::LogInfo("GsfTrajectoryFitter") 
	<< "GsfTrajectoryFitter: predicted tsos not valid after " << myTraj.foundHits()
	<< " hits, discarding candidate!";
	return std::vector<Trajectory>();
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
      myTraj.push(TM(predTsos, currTsos, preciseHit,
		     estimator()->estimate(predTsos, *preciseHit).second,
		     theGeometry->idToLayer(preciseHit->geographicalId() )));
    } else {
      currTsos = predTsos;
      myTraj.push(TM(predTsos, *ihit,0., theGeometry->idToLayer( (*ihit)->geographicalId()) ));
    }
  }
  return std::vector<Trajectory>(1, myTraj);
}

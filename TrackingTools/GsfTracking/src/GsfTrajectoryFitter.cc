#include "TrackingTools/GsfTracking/interface/GsfTrajectoryFitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#include "TrackingTools/TrackFitters/interface/DebugHelpers.h"

GsfTrajectoryFitter::GsfTrajectoryFitter(const Propagator& aPropagator,
                                         const TrajectoryStateUpdator& aUpdator,
                                         const MeasurementEstimator& aEstimator,
                                         const MultiTrajectoryStateMerger& aMerger,
                                         const DetLayerGeometry* detLayerGeometry)
    : thePropagator(aPropagator.clone()),
      theUpdator(aUpdator.clone()),
      theEstimator(aEstimator.clone()),
      theMerger(aMerger.clone()),
      theGeometry(detLayerGeometry) {
  if (!theGeometry)
    theGeometry = &dummyGeometry;
}

GsfTrajectoryFitter::~GsfTrajectoryFitter() {
  delete thePropagator;
  delete theUpdator;
  delete theEstimator;
  delete theMerger;
}

Trajectory GsfTrajectoryFitter::fitOne(const Trajectory& aTraj, fitType type) const {
  if (aTraj.empty())
    return Trajectory();

  TM const& firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());

  return fitOne(aTraj.seed(), aTraj.recHits(), firstTsos, type);
}

Trajectory GsfTrajectoryFitter::fitOne(const TrajectorySeed& aSeed, const RecHitContainer& hits, fitType type) const {
  edm::LogError("GsfTrajectoryFitter") << "GsfTrajectoryFitter::fit(TrajectorySeed, vector<RecHit>) not implemented";

  return Trajectory();
}

Trajectory GsfTrajectoryFitter::fitOne(const TrajectorySeed& aSeed,
                                       const RecHitContainer& hits,
                                       const TrajectoryStateOnSurface& firstPredTsos,
                                       fitType) const {
  if (hits.empty())
    return Trajectory();

  Trajectory myTraj(aSeed, propagator()->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if (!predTsos.isValid()) {
    edm::LogInfo("GsfTrackFitters") << "GsfTrajectoryFitter: predicted tsos of first measurement not valid!";
    return Trajectory();
  }

  TSOS currTsos;
  if (hits.front()->isValid()) {
    auto const& ihit = hits.front();
    //update
    assert((!(ihit)->canImproveWithTrack()) || (nullptr != theHitCloner));
    assert((!(ihit)->canImproveWithTrack()) || (nullptr != dynamic_cast<BaseTrackerRecHit const*>(ihit.get())));
    auto preciseHit = theHitCloner->makeShared(ihit, predTsos);
    dump(*preciseHit, 1, "GsfTrackFitters");
    { currTsos = updator()->update(predTsos, *preciseHit); }
    if (!predTsos.isValid() || !currTsos.isValid()) {
      edm::LogError("InvalidState") << "first hit";
      return Trajectory();
    }
    myTraj.push(TM(predTsos, currTsos, preciseHit, 0., theGeometry->idToLayer(preciseHit->geographicalId())),
                estimator()->estimate(predTsos, *preciseHit).second);
  } else {
    currTsos = predTsos;
    if (!predTsos.isValid()) {
      edm::LogError("InvalidState") << "first invalid hit";
      return Trajectory();
    }
    myTraj.push(TM(predTsos, hits.front(), 0., theGeometry->idToLayer((hits.front())->geographicalId())));
  }

  int hitcounter = 1;
  for (RecHitContainer::const_iterator ihit = hits.begin() + 1; ihit != hits.end(); ihit++) {
    ++hitcounter;

    //
    // temporary protection copied from KFTrajectoryFitter.
    //
    if ((**ihit).isValid() == false && (**ihit).det() == nullptr) {
      LogDebug("GsfTrackFitters") << " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }

    {
      //       TimeMe t(*propTimer,false);
      predTsos = propagator()->propagate(currTsos, (**ihit).det()->surface());
    }
    if (!predTsos.isValid()) {
      if (myTraj.foundHits() >= 3) {
        edm::LogInfo("GsfTrackFitters") << "GsfTrajectoryFitter: predicted tsos not valid! \n"
                                        << "Returning trajectory with " << myTraj.foundHits() << " found hits.";
        return myTraj;
      } else {
        edm::LogInfo("GsfTrackFitters") << "GsfTrajectoryFitter: predicted tsos not valid after " << myTraj.foundHits()
                                        << " hits, discarding candidate!";
        return Trajectory();
      }
    }
    if (merger())
      predTsos = merger()->merge(predTsos);

    if ((**ihit).isValid()) {
      //update
      assert((!(*ihit)->canImproveWithTrack()) || (nullptr != theHitCloner));
      assert((!(*ihit)->canImproveWithTrack()) || (nullptr != dynamic_cast<BaseTrackerRecHit const*>((*ihit).get())));
      if (!predTsos.isValid()) {
        return Trajectory();
      }
      auto preciseHit = theHitCloner->makeShared(*ihit, predTsos);
      dump(*preciseHit, hitcounter, "GsfTrackFitters");
      currTsos = updator()->update(predTsos, *preciseHit);
      if (!predTsos.isValid() || !currTsos.isValid()) {
        edm::LogError("InvalidState") << "inside hit";
        return Trajectory();
      }
      auto chi2 = estimator()->estimate(predTsos, *preciseHit).second;
      myTraj.push(TM(predTsos, currTsos, preciseHit, chi2, theGeometry->idToLayer(preciseHit->geographicalId())));
      LogDebug("GsfTrackFitters") << "added measurement with chi2 " << chi2;
    } else {
      currTsos = predTsos;
      if (!predTsos.isValid()) {
        edm::LogError("InvalidState") << "inside invalid hit";
        return Trajectory();
      }
      myTraj.push(TM(predTsos, *ihit, 0., theGeometry->idToLayer((*ihit)->geographicalId())));
    }
    dump(predTsos, "predTsos", "GsfTrackFitters");
    dump(currTsos, "currTsos", "GsfTrackFitters");
  }
  return myTraj;
}

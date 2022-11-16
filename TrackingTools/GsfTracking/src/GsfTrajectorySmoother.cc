#include "TrackingTools/GsfTracking/interface/GsfTrajectorySmoother.h"

#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrackFitters/interface/DebugHelpers.h"

GsfTrajectorySmoother::GsfTrajectorySmoother(const GsfPropagatorWithMaterial& aPropagator,
                                             const TrajectoryStateUpdator& aUpdator,
                                             const MeasurementEstimator& aEstimator,
                                             const MultiTrajectoryStateMerger& aMerger,
                                             float errorRescaling,
                                             const bool materialBeforeUpdate,
                                             const DetLayerGeometry* detLayerGeometry)
    : theAlongPropagator(nullptr),
      theOppositePropagator(nullptr),
      theGeomPropagator(nullptr),
      theConvolutor(nullptr),
      theUpdator(aUpdator.clone()),
      theEstimator(aEstimator.clone()),
      theMerger(aMerger.clone()),
      theMatBeforeUpdate(materialBeforeUpdate),
      theErrorRescaling(errorRescaling),
      theGeometry(detLayerGeometry) {
  auto p = aPropagator.clone();
  p->setPropagationDirection(alongMomentum);
  theAlongPropagator = p;
  p = aPropagator.clone();
  p->setPropagationDirection(oppositeToMomentum);
  theOppositePropagator = p;
  if (!theMatBeforeUpdate) {
    theGeomPropagator = new GsfPropagatorAdapter(aPropagator.geometricalPropagator());
    theConvolutor = aPropagator.convolutionWithMaterial().clone();
  }

  if (!theGeometry)
    theGeometry = &dummyGeometry;
}

GsfTrajectorySmoother::~GsfTrajectorySmoother() {
  delete theAlongPropagator;
  delete theOppositePropagator;
  delete theGeomPropagator;
  delete theConvolutor;
  delete theUpdator;
  delete theEstimator;
  delete theMerger;
}

Trajectory GsfTrajectorySmoother::trajectory(const Trajectory& aTraj) const {
  if (aTraj.empty())
    return Trajectory();

  const Propagator* usePropagator = theAlongPropagator;
  if (aTraj.direction() == alongMomentum) {
    usePropagator = theOppositePropagator;
  }
  if (not usePropagator) {
    usePropagator = theGeomPropagator;
  }

  Trajectory myTraj(aTraj.seed(), usePropagator->propagationDirection());

  std::vector<TM> const& avtm = aTraj.measurements();

  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);

  if (!predTsos.isValid()) {
    edm::LogInfo("GsfTrackFitters") << "GsfTrajectorySmoother: predicted tsos of last measurement not valid!";
    return Trajectory();
  }
  TSOS currTsos;

  //first smoothed tm is last fitted
  if (avtm.back().recHit()->isValid()) {
    {
      //       TimeMe t(*updateTimer,false);
      currTsos = updator()->update(predTsos, *avtm.back().recHit());
    }

    if (!currTsos.isValid()) {
      edm::LogInfo("GsfTrajectorySmoother") << "GsfTrajectorySmoother: tsos not valid after update!";
      return Trajectory();
    }

    //check validity
    if (!avtm.back().forwardPredictedState().isValid() || !predTsos.isValid() ||
        !avtm.back().updatedState().isValid()) {
      edm::LogError("InvalidState") << "first hit";
      return Trajectory();
    }

    myTraj.push(TM(avtm.back().forwardPredictedState(),
                   predTsos,
                   avtm.back().updatedState(),
                   avtm.back().recHit(),
                   avtm.back().estimate(),
                   theGeometry->idToLayer(avtm.back().recHit()->geographicalId())),
                avtm.back().estimate());
  } else {
    currTsos = predTsos;
    //check validity
    if (!avtm.back().forwardPredictedState().isValid()) {
      edm::LogError("InvalidState") << "first hit on invalid hit";
      return Trajectory();
    }

    myTraj.push(TM(avtm.back().forwardPredictedState(),
                   avtm.back().recHit(),
                   0.,
                   theGeometry->idToLayer(avtm.back().recHit()->geographicalId())));
  }

  TrajectoryStateCombiner combiner;

  int hitcounter = avtm.size() - 1;
  for (std::vector<TM>::const_reverse_iterator itm = avtm.rbegin() + 1; itm < avtm.rend() - 1; ++itm) {
    predTsos = usePropagator->propagate(currTsos, *(*itm).recHit()->surface());
    if (predTsos.isValid() && theConvolutor && theMatBeforeUpdate)
      predTsos = (*theConvolutor)(predTsos, usePropagator->propagationDirection());
    if (!predTsos.isValid()) {
      edm::LogInfo("GsfTrackFitters") << "GsfTrajectorySmoother: predicted tsos not valid!";
      return Trajectory();
    }
    if (theMerger)
      predTsos = theMerger->merge(predTsos);

    if (!predTsos.isValid()) {
      return Trajectory();
    }

    if ((*itm).recHit()->isValid()) {
      //update
      currTsos = updator()->update(predTsos, *(*itm).recHit());

      if (currTsos.isValid() && theConvolutor && !theMatBeforeUpdate)
        currTsos = (*theConvolutor)(currTsos, usePropagator->propagationDirection());
      if (!currTsos.isValid()) {
        edm::LogInfo("GsfTrackFitters") << "GsfTrajectorySmoother: tsos not valid after update / material effects!";
        return Trajectory();
      }
      //3 different possibilities to calculate smoothed state:
      //1: update combined predictions with hit
      //2: combine fwd-prediction with bwd-filter
      //3: combine bwd-prediction with fwd-filter
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());
      if (!combTsos.isValid()) {
        LogDebug("GsfTrackFitters") << "KFTrajectorySmoother: combined tsos not valid!\n"
                                    << "pred Tsos pos: " << predTsos.globalPosition() << "\n"
                                    << "pred Tsos mom: " << predTsos.globalMomentum() << "\n"
                                    << "TrackingRecHit: "
                                    << (*itm).recHit()->surface()->toGlobal((*itm).recHit()->localPosition()) << "\n";
        return Trajectory();
      }

      TSOS smooTsos = combiner((*itm).updatedState(), predTsos);

      if (!smooTsos.isValid()) {
        LogDebug("GsfTrackFitters") << "KFTrajectorySmoother: smoothed tsos not valid!";
        return Trajectory();
      }

      if (!(*itm).forwardPredictedState().isValid() || !predTsos.isValid() || !smooTsos.isValid()) {
        edm::LogError("InvalidState") << "inside hits with combination.";
        return Trajectory();
      }

      auto chi2 = estimator()->estimate(combTsos, *(*itm).recHit()).second;
      myTraj.push(TM((*itm).forwardPredictedState(),
                     predTsos,
                     smooTsos,
                     (*itm).recHit(),
                     chi2,
                     theGeometry->idToLayer((*itm).recHit()->geographicalId())),
                  (*itm).estimate());
      LogDebug("GsfTrackFitters") << "added measurement #" << hitcounter-- << " with chi2 " << chi2;
      dump(predTsos, "predTsos", "GsfTrackFitters");
      dump(smooTsos, "smooTsos", "GsfTrackFitters");

    } else {
      currTsos = predTsos;
      TSOS combTsos = combiner(predTsos, (*itm).forwardPredictedState());

      if (!combTsos.isValid()) {
        LogDebug("GsfTrackFitters") << "KFTrajectorySmoother: combined tsos not valid!";
        return Trajectory();
      }

      if (!(*itm).forwardPredictedState().isValid() || !predTsos.isValid() || !combTsos.isValid()) {
        edm::LogError("InvalidState") << "inside hits with invalid rechit.";
        return Trajectory();
      }

      myTraj.push(TM((*itm).forwardPredictedState(),
                     predTsos,
                     combTsos,
                     (*itm).recHit(),
                     0.,
                     theGeometry->idToLayer((*itm).recHit()->geographicalId())));
      LogDebug("GsfTrackFitters") << "added invalid measurement #" << hitcounter--;
      dump(predTsos, "predTsos", "GsfTrackFitters");
      dump(combTsos, "smooTsos", "GsfTrackFitters");
    }
    if (theMerger)
      currTsos = theMerger->merge(currTsos);

    if (!currTsos.isValid()) {
      return Trajectory();
    }
    dump(currTsos, "currTsos", "GsfTrackFitters");
  }

  //last smoothed tm is last filtered
  predTsos = usePropagator->propagate(currTsos, *avtm.front().recHit()->surface());
  if (predTsos.isValid() && theConvolutor && theMatBeforeUpdate)
    predTsos = (*theConvolutor)(predTsos, usePropagator->propagationDirection());
  if (!predTsos.isValid()) {
    edm::LogInfo("GsfTrackFitters") << "GsfTrajectorySmoother: predicted tsos not valid!";
    return Trajectory();
  }
  if (theMerger)
    predTsos = theMerger->merge(predTsos);

  if (avtm.front().recHit()->isValid()) {
    //update
    currTsos = updator()->update(predTsos, *avtm.front().recHit());
    if (currTsos.isValid() && theConvolutor && !theMatBeforeUpdate)
      currTsos = (*theConvolutor)(currTsos, usePropagator->propagationDirection());
    if (!currTsos.isValid()) {
      edm::LogInfo("GsfTrackFitters") << "GsfTrajectorySmoother: tsos not valid after update / material effects!";
      return Trajectory();
    }

    if (!avtm.front().forwardPredictedState().isValid() || !predTsos.isValid() || !currTsos.isValid()) {
      edm::LogError("InvalidState") << "last hit";
      return Trajectory();
    }

    auto chi2 = estimator()->estimate(predTsos, *avtm.front().recHit()).second;
    myTraj.push(TM(avtm.front().forwardPredictedState(),
                   predTsos,
                   currTsos,
                   avtm.front().recHit(),
                   chi2,
                   theGeometry->idToLayer(avtm.front().recHit()->geographicalId())),
                avtm.front().estimate());
    LogDebug("GsfTrackFitters") << "added measurement #" << hitcounter-- << " with chi2 " << chi2;
    dump(predTsos, "predTsos", "GsfTrackFitters");
    dump(currTsos, "smooTsos", "GsfTrackFitters");

  } else {
    if (!avtm.front().forwardPredictedState().isValid()) {
      edm::LogError("InvalidState") << "last invalid hit";
      return Trajectory();
    }
    myTraj.push(TM(avtm.front().forwardPredictedState(),
                   avtm.front().recHit(),
                   0.,
                   theGeometry->idToLayer(avtm.front().recHit()->geographicalId())));
    LogDebug("GsfTrackFitters") << "added invalid measurement #" << hitcounter--;
  }

  return myTraj;
}

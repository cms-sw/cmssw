#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/DebugHelpers.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "FWCore/Utilities/interface/Likely.h"

KFTrajectorySmoother::~KFTrajectorySmoother() {
  delete theAlongPropagator;
  delete theOppositePropagator;
  delete theUpdator;
  delete theEstimator;
}

Trajectory KFTrajectorySmoother::trajectory(const Trajectory& aTraj) const {
  if (aTraj.empty())
    return Trajectory();

  const Propagator* usePropagator = theAlongPropagator;
  if (aTraj.direction() == alongMomentum) {
    usePropagator = theOppositePropagator;
  }

  const std::vector<TM>& avtm = aTraj.measurements();

#ifdef EDM_ML_DEBUG
  LogDebug("TrackFitters") << "KFTrajectorySmoother::trajectories starting with " << avtm.size() << " HITS\n";
  for (unsigned int j = 0; j < avtm.size(); j++) {
    if (avtm[j].recHit()->det())
      LogTrace("TrackFitters") << "hit #:" << j + 1 << " rawId=" << avtm[j].recHit()->det()->geographicalId().rawId()
                               << " validity=" << avtm[j].recHit()->isValid();
    else
      LogTrace("TrackFitters") << "hit #:" << j + 1 << " Hit with no Det information";
  }
#endif  // EDM_ML_DEBUG

  TrajectoryStateCombiner combiner;
  bool retry = false;
  auto start = avtm.rbegin();

  do {
    auto hitSize = avtm.rend() - start;
    if UNLIKELY (hitSize < minHits_) {
      LogDebug("TrackFitters") << " killing trajectory"
                               << "\n";
      return Trajectory();
    }
    Trajectory ret(aTraj.seed(), usePropagator->propagationDirection());
    Trajectory& myTraj = ret;
    myTraj.reserve(hitSize);
    retry = false;

    TSOS predTsos = (*start).forwardPredictedState();
    predTsos.rescaleError(theErrorRescaling);
    TSOS currTsos;

    auto hitCounter = hitSize;
    for (std::vector<TM>::const_reverse_iterator itm = start; itm != (avtm.rend()); ++itm, --hitCounter) {
      TransientTrackingRecHit::ConstRecHitPointer hit = itm->recHit();

      //check surface just for safety: should never be ==0 because they are skipped in the fitter
      // if UNLIKELY(hit->det() == nullptr) continue;
      if UNLIKELY (hit->surface() == nullptr) {
        LogDebug("TrackFitters") << " Error: invalid hit with no GeomDet attached .... skipping";
        continue;
      }

      if (itm != start)  //no propagation needed for first smoothed (==last fitted) hit
        predTsos = usePropagator->propagate(currTsos, *(hit->surface()));

      if UNLIKELY (!predTsos.isValid()) {
        LogDebug("TrackFitters") << "KFTrajectorySmoother: predicted tsos not valid!";
        LogDebug("TrackFitters") << " retry with last hit removed"
                                 << "\n";
        LogDebug("TrackFitters")
            // std::cout
            << "tsos not valid " << currTsos.globalMomentum().perp() << ' ' << hitSize << ' ' << hitCounter << ' '
            << int(hit->geographicalId()) << ' ' << hit->surface()->position().perp() << ' ' << hit->surface()->eta()
            << ' ' << hit->surface()->phi() << std::endl;
        start++;
        retry = true;
        break;
      }

      if (hit->isValid()) {
        TSOS combTsos, smooTsos;

        //3 different possibilities to calculate smoothed state:
        //1: update combined predictions with hit
        //2: combine fwd-prediction with bwd-filter
        //3: combine bwd-prediction with fwd-filter

        //combTsos is the predicted state with N-1 hits information. this means:
        //forward predicted state for first smoothed (last fitted) hit
        //backward predicted state for last smoothed (first fitted) hit
        //combination of forward and backward predictions for other hits
        if (itm == start)
          combTsos = itm->forwardPredictedState();
        else if (hitCounter == 1)
          combTsos = predTsos;
        else
          combTsos = combiner(predTsos, itm->forwardPredictedState());

        if UNLIKELY (!combTsos.isValid()) {
          LogDebug("TrackFitters") << "KFTrajectorySmoother: combined tsos not valid!\n"
                                   << "pred Tsos pos: " << predTsos.globalPosition() << "\n"
                                   << "pred Tsos mom: " << predTsos.globalMomentum() << "\n"
                                   << "TrackingRecHit: " << hit->surface()->toGlobal(hit->localPosition()) << "\n";
          start++;
          retry = true;
          break;
        }

        assert((hit->geographicalId() != 0U) | (!hit->canImproveWithTrack()));
        assert(hit->surface() != nullptr);
        assert((!(hit)->canImproveWithTrack()) | (nullptr != theHitCloner));
        assert((!(hit)->canImproveWithTrack()) | (nullptr != dynamic_cast<BaseTrackerRecHit const*>(hit.get())));
        auto preciseHit = theHitCloner->makeShared(hit, combTsos);
        assert(preciseHit->isValid());
        assert((preciseHit->geographicalId() != 0U) | (!preciseHit->canImproveWithTrack()));
        assert(preciseHit->surface() != nullptr);

        dump(*hit, hitCounter, "TrackFitters");

        if UNLIKELY (!preciseHit->isValid()) {
          LogTrace("TrackFitters") << "THE Precise HIT IS NOT VALID: using currTsos = predTsos"
                                   << "\n";
          currTsos = predTsos;
          myTraj.push(TM(predTsos, hit, 0, theGeometry->idToLayer(hit->geographicalId())));
        } else {
          LogTrace("TrackFitters") << "THE Precise HIT IS VALID: updating currTsos"
                                   << "\n";

          //update backward predicted tsos with the hit
          currTsos = updator()->update(predTsos, *preciseHit);
          if UNLIKELY (!currTsos.isValid()) {
            currTsos = predTsos;
            edm::LogWarning("KFSmoother_UpdateFailed")
                << "Failed updating state with hit. Rolling back to non-updated state.\n"
                << "State: " << predTsos << "Hit local pos:  " << hit->localPosition() << "\n"
                << "Hit local err:  " << hit->localPositionError() << "\n"
                << "Hit global pos: " << hit->globalPosition() << "\n"
                << "Hit global err: " << hit->globalPositionError().matrix() << "\n";
          }

          //smooTsos updates the N-1 hits prediction with the hit
          if (itm == start)
            smooTsos = itm->updatedState();
          else if (hitCounter == 1)
            smooTsos = currTsos;
          else
            smooTsos = combiner(itm->forwardPredictedState(), currTsos);

          if UNLIKELY (!smooTsos.isValid()) {
            LogDebug("TrackFitters") << "KFTrajectorySmoother: smoothed tsos not valid!";
            start++;
            retry = true;
            break;
          }

          double estimate;
          if (itm != start)
            estimate = estimator()->estimate(combTsos, *preciseHit).second;  //correct?
          else
            estimate = itm->estimate();

          LogTrace("TrackFitters") << "predTsos !"
                                   << "\n"
                                   << predTsos << " with local position " << predTsos.localPosition() << "\n\n"
                                   << "currTsos !"
                                   << "\n"
                                   << currTsos << "\n"
                                   << " with local position " << currTsos.localPosition() << "\n\n"
                                   << "smooTsos !"
                                   << "\n"
                                   << smooTsos << " with local position " << smooTsos.localPosition() << "\n\n"
                                   << "smoothing estimate (with combTSOS)=" << estimate << "\n"
                                   << "filtering estimate=" << itm->estimate() << "\n";

          //check for valid hits with no det (refitter with constraints)
          if (preciseHit->det())
            myTraj.push(TM(itm->forwardPredictedState(),
                           predTsos,
                           smooTsos,
                           preciseHit,
                           estimate,
                           theGeometry->idToLayer(preciseHit->geographicalId())),
                        estimator()->estimate(predTsos, *preciseHit).second);
          else
            myTraj.push(TM(itm->forwardPredictedState(), predTsos, smooTsos, preciseHit, estimate),
                        estimator()->estimate(predTsos, *preciseHit).second);
          //itm->estimate());
        }
      } else {
        LogDebug("TrackFitters") << "----------------- HIT #" << hitCounter << " (INVALID)-----------------------";

        //no update
        currTsos = predTsos;
        TSOS combTsos;
        if (itm == start)
          combTsos = itm->forwardPredictedState();
        else if (hitCounter == 1)
          combTsos = predTsos;
        else
          combTsos = combiner(predTsos, itm->forwardPredictedState());

        if UNLIKELY (!combTsos.isValid()) {
          LogDebug("TrackFitters") << "KFTrajectorySmoother: combined tsos not valid!";
          return Trajectory();
        }
        assert((hit->det() == nullptr) || hit->geographicalId() != 0U);
        if (hit->det())
          myTraj.push(TM(
              itm->forwardPredictedState(), predTsos, combTsos, hit, 0, theGeometry->idToLayer(hit->geographicalId())));
        else
          myTraj.push(TM(itm->forwardPredictedState(), predTsos, combTsos, hit, 0));
      }
    }  // for loop

    if (!retry)
      return ret;
  } while (true);

  return Trajectory();
}

#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/DetGroup.h"

#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <algorithm>
#include <cstdlib>

using namespace std;

namespace {
  typedef GeometricSearchDet::DetWithState DetWithState;
  inline void addInvalidMeas(std::vector<TrajectoryMeasurement>& result,
                             const TrajectoryStateOnSurface& ts,
                             const GeomDet& det,
                             const DetLayer& layer) {
    result.emplace_back(ts, std::make_shared<InvalidTrackingRecHit>(det, TrackingRecHit::missing), 0.F, &layer);
  }

  // layerIsCompatibleWithTSOS:
  //  - enforce minimum requirements of compatibility between DetLayer and TrajectoryStateOnSurface
  //    (e.g. global position of TSOS within detector volume)
  //  - current implementation does not make use of layer argument,
  //    see below for an example of how to use it
  //      #include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
  //       [..]
  //      using namespace GeomDetEnumerators;
  //      auto const& det = layer.subDetector();
  //      if (det == PixelBarrel or det == PixelEndcap) {
  //        return xa < 2e1 and ya < 2e1 and za < 6e1;
  //      }
  bool layerIsCompatibleWithTSOS(DetLayer const& /* layer */, TrajectoryStateOnSurface const& tsos) {
    auto const& gp = tsos.globalPosition();
    auto const xa = std::abs(gp.x());
    auto const ya = std::abs(gp.y());
    auto const za = std::abs(gp.z());
    return xa < 1e3 and ya < 1e3 and za < 12e2;
  }

  /** The std::vector<DetWithState> passed to this method should not be empty.
   *  In case of no compatible dets the result should be either an empty container if 
   *  the det is itself incompatible, or a container with one invalid measurement
   *  on the det surface. The method does not have enough information to do
   *  this efficiently, so it should be done by the caller, or an exception will
   *  be thrown (DetLogicError).
   */
  inline std::vector<TrajectoryMeasurement> get(MeasurementDetSystem const& detSystem,
                                                MeasurementTrackerEvent const& data,
                                                const DetLayer& layer,
                                                std::vector<DetWithState> const& compatDets,
                                                const TrajectoryStateOnSurface& ts,
                                                const Propagator& prop,
                                                const MeasurementEstimator& est) {
    std::vector<TrajectoryMeasurement> result;
    typedef TrajectoryMeasurement TM;

    tracking::TempMeasurements tmps;

    for (auto const& ds : compatDets) {
      if (not layerIsCompatibleWithTSOS(layer, ds.second))
        continue;

      MeasurementDetWithData mdet = detSystem.idToDet(ds.first->geographicalId(), data);
      if UNLIKELY (mdet.isNull()) {
        throw MeasurementDetException("MeasurementDet not found");
      }

      if (mdet.measurements(ds.second, est, tmps))
        for (std::size_t i = 0; i != tmps.size(); ++i)
          result.emplace_back(ds.second, std::move(tmps.hits[i]), tmps.distances[i], &layer);
      tmps.clear();
    }
    // WARNING: we might end up with more than one invalid hit of type 'inactive' in result
    // to be fixed in order to avoid usless double traj candidates.

    // sort the final result
    if (result.size() > 1) {
      sort(result.begin(), result.end(), TrajMeasLessEstim());
    }

    if (!result.empty()) {
      // invalidMeas on Det of most compatible hit
      addInvalidMeas(result, result.front().predictedState(), *(result.front().recHit()->det()), layer);
    } else {
      // invalid state on first compatible Det
      addInvalidMeas(result, compatDets.front().second, *(compatDets.front().first), layer);
    }

    return result;
  }

  void addInvalidMeas(vector<TrajectoryMeasurement>& measVec, const DetGroup& group, const DetLayer& layer) {
    if (!measVec.empty()) {
      // invalidMeas on Det of most compatible hit
      auto const& ts = measVec.front().predictedState();
      auto toll = measVec.front().recHitR().det()->surface().bounds().significanceInside(
          ts.localPosition(), ts.localError().positionError());
      measVec.emplace_back(
          measVec.front().predictedState(),
          std::make_shared<InvalidTrackingRecHit>(*measVec.front().recHitR().det(), TrackingRecHit::missing),
          toll,
          &layer);
    } else if (!group.empty()) {
      // invalid state on first compatible Det
      auto const& ts = group.front().trajectoryState();
      auto toll = group.front().det()->surface().bounds().significanceInside(ts.localPosition(),
                                                                             ts.localError().positionError());
      measVec.emplace_back(group.front().trajectoryState(),
                           std::make_shared<InvalidTrackingRecHit>(*group.front().det(), TrackingRecHit::missing),
                           toll,
                           &layer);
    }
  }

}  // namespace

// return just valid hits, no sorting (for seeding mostly)
std::vector<BaseTrackerRecHit*> LayerMeasurements::recHits(const DetLayer& layer,
                                                           const TrajectoryStateOnSurface& startingState,
                                                           const Propagator& prop,
                                                           const MeasurementEstimator& est) const {
  std::vector<BaseTrackerRecHit*> result;
  auto const& compatDets = layer.compatibleDets(startingState, prop, est);
  if (compatDets.empty())
    return result;
  for (auto const& ds : compatDets) {
    if (not layerIsCompatibleWithTSOS(layer, ds.second))
      continue;

    auto mdet = detSystem_.idToDet(ds.first->geographicalId(), data_);
    mdet.recHits(result, ds.second, est);
  }
  return result;
}

vector<TrajectoryMeasurement> LayerMeasurements::measurements(const DetLayer& layer,
                                                              const TrajectoryStateOnSurface& startingState,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est) const {
  typedef DetLayer::DetWithState DetWithState;

  vector<DetWithState> const& compatDets = layer.compatibleDets(startingState, prop, est);

  if (!compatDets.empty())
    return get(detSystem_, data_, layer, compatDets, startingState, prop, est);

  vector<TrajectoryMeasurement> result;
  pair<bool, TrajectoryStateOnSurface> compat = layer.compatible(startingState, prop, est);

  if (compat.first and layerIsCompatibleWithTSOS(layer, compat.second)) {
    result.push_back(
        TrajectoryMeasurement(compat.second,
                              std::make_shared<InvalidTrackingRecHitNoDet>(layer.surface(), TrackingRecHit::inactive),
                              0.F,
                              &layer));
    LogDebug("LayerMeasurements") << "adding a missing hit.";
  } else
    LogDebug("LayerMeasurements") << "adding not measurement.";

  return result;
}

vector<TrajectoryMeasurementGroup> LayerMeasurements::groupedMeasurements(const DetLayer& layer,
                                                                          const TrajectoryStateOnSurface& startingState,
                                                                          const Propagator& prop,
                                                                          const MeasurementEstimator& est) const {
  vector<TrajectoryMeasurementGroup> result;

  vector<DetGroup>&& groups = layer.groupedCompatibleDets(startingState, prop, est);
  result.reserve(groups.size());

  tracking::TempMeasurements tmps;
  for (auto& grp : groups) {
    if (grp.empty())
      continue;

    vector<TrajectoryMeasurement> tmpVec;
    for (auto const& det : grp) {
      auto const& detTS = det.trajectoryState();

      if (not layerIsCompatibleWithTSOS(layer, detTS))
        continue;

      MeasurementDetWithData mdet = detSystem_.idToDet(det.det()->geographicalId(), data_);
      if (mdet.isNull()) {
        throw MeasurementDetException("MeasurementDet not found");
      }
      if (mdet.measurements(detTS, est, tmps))
        for (std::size_t i = 0; i != tmps.size(); ++i)
          tmpVec.emplace_back(detTS, std::move(tmps.hits[i]), tmps.distances[i], &layer);
      tmps.clear();
    }

    // sort the final result
    LogDebug("LayerMeasurements") << "Sorting " << tmpVec.size() << " measurements in this grp.";
    sort(tmpVec.begin(), tmpVec.end(), TrajMeasLessEstim());
    addInvalidMeas(tmpVec, grp, layer);
    result.emplace_back(std::move(tmpVec), std::move(grp));
  }

  // if the result is empty check if the layer is compatible (for invalid measurement)
  if (result.empty()) {
    pair<bool, TrajectoryStateOnSurface> compat = layer.compatible(startingState, prop, est);
    if (compat.first and layerIsCompatibleWithTSOS(layer, compat.second)) {
      vector<TrajectoryMeasurement> tmVec;
      tmVec.emplace_back(compat.second,
                         std::make_shared<InvalidTrackingRecHitNoDet>(layer.surface(), TrackingRecHit::inactive),
                         0.F,
                         &layer);
      result.emplace_back(std::move(tmVec), DetGroup());
    }
  }
  return result;
}

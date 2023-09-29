#include "TrackingTools/GsfTracking/interface/GsfMultiStateUpdator.h"
#include "TrackingTools/GsfTools/interface/GetComponents.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "TrackingTools/GsfTracking/interface/PosteriorWeightsCalculator.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrajectoryStateOnSurface GsfMultiStateUpdator::update(const TrajectoryStateOnSurface& tsos,
                                                      const TrackingRecHit& aRecHit) const {
  if (!tsos.isValid()) {
    edm::LogError("GsfMultiStateUpdator") << "Trying to update trajectory state with invalid TSOS! ";
    return TrajectoryStateOnSurface();
  }

  GetComponents comps(tsos);
  auto const& predictedComponents = comps();
  if (predictedComponents.empty()) {
    edm::LogError("GsfMultiStateUpdator") << "Trying to update trajectory state with zero components! ";
    return TrajectoryStateOnSurface();
  }

  auto&& weights = PosteriorWeightsCalculator(predictedComponents).weights(aRecHit);
  if (weights.empty()) {
    edm::LogError("GsfMultiStateUpdator") << " no weights could be retreived. invalid updated state !.";
    return TrajectoryStateOnSurface();
  }

  MultiTrajectoryStateAssembler result;

  int i = 0;
  for (auto const& tsosI : predictedComponents) {
    TrajectoryStateOnSurface updatedTSOS = KFUpdator().update(tsosI, aRecHit);

    if (double det;
        updatedTSOS.isValid() && updatedTSOS.localError().valid() && updatedTSOS.localError().posDef() &&
        (det = 0., updatedTSOS.curvilinearError().matrix().Sub<AlgebraicSymMatrix22>(0, 0).Det(det) && det > 0) &&
        (det = 0., updatedTSOS.curvilinearError().matrix().Sub<AlgebraicSymMatrix33>(0, 0).Det(det) && det > 0) &&
        (det = 0., updatedTSOS.curvilinearError().matrix().Sub<AlgebraicSymMatrix44>(0, 0).Det(det) && det > 0) &&
        (det = 0., updatedTSOS.curvilinearError().matrix().Det2(det) && det > 0)) {
      result.addState(TrajectoryStateOnSurface(weights[i],
                                               updatedTSOS.localParameters(),
                                               updatedTSOS.localError(),
                                               updatedTSOS.surface(),
                                               &(tsos.globalParameters().magneticField()),
                                               tsosI.surfaceSide()));
    } else {
      edm::LogError("GsfMultiStateUpdator") << "KF updated state " << i << " is invalid. skipping.";
    }
    ++i;
  }

  return result.combinedState();
}

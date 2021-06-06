#include <memory>
#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"

#include "RecoVertex/GhostTrackFitter/interface/AnnealingGhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/PositiveSideGhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/KalmanGhostTrackUpdater.h"

using namespace reco;

GhostTrackFitter::GhostTrackFitter() {
  fitter = std::make_unique<AnnealingGhostTrackFitter>();
  updater = std::make_unique<KalmanGhostTrackUpdater>();
}

GhostTrackFitter::~GhostTrackFitter() {}

GhostTrack GhostTrackFitter::fit(const GlobalPoint &priorPosition,
                                 const GlobalError &priorError,
                                 const GlobalVector &direction,
                                 double coneRadius,
                                 const std::vector<TransientTrack> &tracks) const {
  GhostTrackPrediction prior(priorPosition, priorError, direction, coneRadius);
  return fit(prior, priorPosition, tracks);
}

GhostTrack GhostTrackFitter::fit(const GlobalPoint &priorPosition,
                                 const GlobalError &priorError,
                                 const GlobalVector &direction,
                                 const GlobalError &directionError,
                                 const std::vector<TransientTrack> &tracks) const {
  GhostTrackPrediction prior(priorPosition, priorError, direction, directionError);
  return fit(prior, priorPosition, tracks);
}

GhostTrack GhostTrackFitter::fit(const GhostTrackPrediction &prior,
                                 const GlobalPoint &origin,
                                 const std::vector<TransientTrack> &tracks) const {
  double offset = prior.lambda(origin);

  std::vector<GhostTrackState> states;
  for (std::vector<TransientTrack>::const_iterator iter = tracks.begin(); iter != tracks.end(); ++iter) {
    GhostTrackState state(*iter);
    state.linearize(prior, true, offset);
    states.push_back(state);
  }

  PositiveSideGhostTrackFitter actualFitter(origin, *fitter);
  return fit(actualFitter, prior, states);
}

GhostTrack GhostTrackFitter::fit(const GhostTrackPrediction &prior, const std::vector<TransientTrack> &tracks) const {
  std::vector<GhostTrackState> states;
  for (std::vector<TransientTrack>::const_iterator iter = tracks.begin(); iter != tracks.end(); ++iter) {
    GhostTrackState state(*iter);
    state.linearize(prior, true);
    states.push_back(state);
  }

  return fit(*fitter, prior, states);
}

GhostTrack GhostTrackFitter::fit(FitterImpl &fitterImpl,
                                 const GhostTrackPrediction &prior,
                                 const std::vector<GhostTrackState> &states_) const {
  std::vector<GhostTrackState> states = states_;

  double ndof, chi2;
  GhostTrackPrediction pred = fitterImpl.fit(*updater, prior, states, ndof, chi2);

  GhostTrack result(prior, pred, states, ndof, chi2);

  return result;
}

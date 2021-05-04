#include <memory>

#include <vector>

#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/AnnealingGhostTrackFitter.h"

using namespace reco;

AnnealingGhostTrackFitter::AnnealingGhostTrackFitter() : firstStep(true) {
  annealing = std::make_unique<GeometricAnnealing>(3.0, 64.0, 0.25);
}

void AnnealingGhostTrackFitter::postFit(const GhostTrackFitter::PredictionUpdater &updater,
                                        const GhostTrackPrediction &pred,
                                        std::vector<GhostTrackState> &states) {
  for (std::vector<GhostTrackState>::iterator state = states.begin(); state != states.end(); ++state) {
    if (!state->isValid())
      continue;

    double ndof, chi2;
    updater.contribution(pred, *state, ndof, chi2);
    if (ndof == 0. || firstStep)
      continue;

    state->setWeight(annealing->weight(chi2));
  }

  if (firstStep)
    firstStep = false;
  else
    annealing->anneal();
}

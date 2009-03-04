#include <vector>

#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/AnnealingGhostTrackFitter.h"

using namespace reco;

namespace {
	static inline double sqr(double arg) { return arg * arg; }
}

AnnealingGhostTrackFitter::AnnealingGhostTrackFitter()
{
	annealing.reset(new GeometricAnnealing(2.5, 16.0, 0.25));
}

void AnnealingGhostTrackFitter::postFit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &pred,
			std::vector<GhostTrackState> &states)
{
	for(std::vector<GhostTrackState>::iterator state = 
		states.begin(); state != states.end(); ++state) {

		if (!state->isValid())
			continue;

		double ndof, chi2;
		updater.contribution(pred, *state, ndof, chi2);
		if (ndof == 0. || firstStep)
			continue;

		double weight = annealing->weight(chi2 / state->weight());
// std::cout << "chi2 = " << chi2 << ", weight = " << weight << std::endl;

		state->setWeight(weight);
	}

	if (firstStep)
		firstStep = false;
	else
		annealing->anneal();
}

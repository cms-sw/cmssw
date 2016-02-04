#include <cmath>
#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/PositiveSideGhostTrackFitter.h"

using namespace reco;

GhostTrackPrediction PositiveSideGhostTrackFitter::fit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &prior,
			std::vector<GhostTrackState> &states,
			double &ndof, double &chi2)
{
	double rho = prior.rho();
	for(unsigned int i = 0; i < states.size(); i++) {
		GhostTrackState &state = states[i];
		state.linearize(prior, true, .5 / rho);
	}

	GhostTrackPrediction pred =
			actualFitter_->fit(updater, prior, states, ndof, chi2);

	double origin = pred.lambda(origin_);
	bool done = true;
	for(unsigned int i = 0; i < states.size(); i++) {
		GhostTrackState &state = states[i];
		double lambda = state.lambda();
		if (lambda < origin && (origin - lambda) < 3.5) {
			GhostTrackState testState = state;
			testState.linearize(pred, 2. * origin - lambda);
			double ndof, chi2;

			updater.contribution(prior, testState, ndof, chi2, true);
			if (ndof > 0. && chi2 < 10.) {
				state = testState;
				if (state.weight() != 1.)
					state.setWeight(3.);
				done = false;
			}
		}
	}

	if (!done) {
		for(unsigned int i = 0; i < states.size(); i++) {
			GhostTrackState &state = states[i];
			double lambda = state.lambda();
			if (state.weight() != 1. && lambda < origin) {
				double weight =
					std::exp(10. * (origin - lambda) - .1);
				state.setWeight(
					std::min(state.weight(), weight));
			}
		}

		pred = actualFitter_->fit(updater, prior, states, ndof, chi2);
	}

	return pred;
}

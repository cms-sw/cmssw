#include <vector>

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/SequentialGhostTrackFitter.h"

using namespace reco;

namespace {
	static inline double sqr(double arg) { return arg * arg; }
}

SequentialGhostTrackFitter::SequentialGhostTrackFitter() :
	maxIteration(15),
	minDeltaR(0.0015),
	minDistance(0.002),
	weightThreshold(0.001)
{
}

bool SequentialGhostTrackFitter::stable(
				const GhostTrackPrediction &before,
				const GhostTrackPrediction &after) const
{
	return (sqr(after.sz() - before.sz()) +
	        sqr(after.ip() - before.ip()) < sqr(minDistance) &&
	        sqr(after.eta() - before.eta()) +
	        sqr(after.phi() - before.phi()) < sqr(minDeltaR));
}

GhostTrackPrediction SequentialGhostTrackFitter::fit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &prior,
			std::vector<GhostTrackState> &states,
			double &ndof, double &chi2)
{
	GhostTrackPrediction pred, lastPred = prior;

	reset();

	ndof = 0.;
	chi2 = 0.;

	unsigned int iteration = 0;
	for(;;) {
		pred = prior;

		if (states.begin() == states.end())
			break;

		if (iteration > 0) {
			for(unsigned int i = 0; i < states.size(); i++) {
				GhostTrackState &state = states[i];
				state.linearize(lastPred);
			}
		}

		ndof = 0.; // prior gives us an initial ndof
		chi2 = 0.;

		for(std::vector<GhostTrackState>::const_iterator state = 
			states.begin(); state != states.end(); ++state) {

			if (state->isValid() &&
			    state->weight() > weightThreshold)
				pred = updater.update(pred, *state,
				                      ndof, chi2);
		}

		if (++iteration >= maxIteration || stable(lastPred, pred))
			break;

		postFit(updater, pred, states);

		lastPred = pred;
	}

	return pred;
}

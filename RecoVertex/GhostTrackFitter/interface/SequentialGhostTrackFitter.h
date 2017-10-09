#ifndef RecoBTag_SequentialGhostTrackFitter_h
#define RecoBTag_SequentialGhostTrackFitter_h

#include <vector>

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"

namespace reco {

class GhostTrackPrediction;
class GhostTrackState;

class SequentialGhostTrackFitter : public GhostTrackFitter::FitterImpl {
    public:
	SequentialGhostTrackFitter();
	~SequentialGhostTrackFitter() {}

	GhostTrackPrediction fit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &prior,
			std::vector<GhostTrackState> &states,
			double &ndof, double &chi2);

    protected:
	virtual bool stable(const GhostTrackPrediction &before,
	                    const GhostTrackPrediction &after) const;
	virtual void reset() {}
	virtual void postFit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &pred,
			std::vector<GhostTrackState> &states) {}

    private:
	virtual FitterImpl *clone() const
	{ return new SequentialGhostTrackFitter(*this); }

	unsigned int	maxIteration;
	double		minDeltaR;
	double		minDistance;
	double		weightThreshold;
};

}

#endif // RecoBTag_SequentialGhostTrackFitter_h

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
	~SequentialGhostTrackFitter() override {}

	GhostTrackPrediction fit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &prior,
			std::vector<GhostTrackState> &states,
			double &ndof, double &chi2) override;

    protected:
	virtual bool stable(const GhostTrackPrediction &before,
	                    const GhostTrackPrediction &after) const;
	virtual void reset() {}
	virtual void postFit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &pred,
			std::vector<GhostTrackState> &states) {}

    private:
	FitterImpl *clone() const override
	{ return new SequentialGhostTrackFitter(*this); }

	unsigned int	maxIteration;
	double		minDeltaR;
	double		minDistance;
	double		weightThreshold;
};

}

#endif // RecoBTag_SequentialGhostTrackFitter_h

#ifndef RecoBTag_KalmanGhostTrackUpdater_h
#define RecoBTag_KalmanGhostTrackUpdater_h

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"

namespace reco {

class GhostTrackPrediction;
class GhostTrackState;

class KalmanGhostTrackUpdater : public GhostTrackFitter::PredictionUpdater {
    public:
	~KalmanGhostTrackUpdater() override {}

	KalmanGhostTrackUpdater *clone() const override
	{ return new KalmanGhostTrackUpdater(*this); }

	GhostTrackPrediction update(const GhostTrackPrediction &pred,
	                            const GhostTrackState &state,
	                            double &ndof, double &chi2) const override;

	void contribution(const GhostTrackPrediction &pred,
	                  const GhostTrackState &state,
	                  double &ndof, double &chi2,
	                  bool withPredError = false) const override;
};

}

#endif // RecoBTag_KalmanGhostTrackUpdater_h

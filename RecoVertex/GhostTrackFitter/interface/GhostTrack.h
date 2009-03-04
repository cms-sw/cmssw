#ifndef RecoBTag_GhostTrack_h
#define RecoBTag_GhostTrack_h

#include <vector>

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

namespace reco {

class GhostTrack {
    public:
	GhostTrack(const GhostTrackPrediction &prior,
	           const GhostTrackPrediction &prediction,
	           const std::vector<GhostTrackState> &states,
	           double ndof, double chi2) :
		prediction_(prediction), prior_(prior), states_(states),
		ndof_(ndof), chi2_(chi2)
	{}

	const GhostTrackPrediction &prediction() const { return prediction_; }
	const GhostTrackPrediction &prior() const { return prior_; }

	const std::vector<GhostTrackState> &states() const { return states_; }
	double ndof() const { return ndof_; }
	double chi2() const { return chi2_; }

    private:
	GhostTrackPrediction		prediction_;
	GhostTrackPrediction		prior_;
	std::vector<GhostTrackState>	states_;
	double				ndof_;
	double				chi2_;
};

}

#endif // RecoBTag_GhostTrack_h

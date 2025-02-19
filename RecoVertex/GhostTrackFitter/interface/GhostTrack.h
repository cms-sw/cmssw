#ifndef RecoBTag_GhostTrack_h
#define RecoBTag_GhostTrack_h

#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

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

	GhostTrack(const GhostTrackPrediction &prior,
	           const GhostTrackPrediction &prediction,
	           const std::vector<TransientTrack> &tracks,
	           double ndof, double chi2,
	           const std::vector<float> &weights = std::vector<float>(),
	           const GlobalPoint &origin = GlobalPoint(),
	           bool withOrigin = false);

	GhostTrack(const Track &ghostTrack,
	           const std::vector<TransientTrack> &tracks,
	           const std::vector<float> &weights = std::vector<float>(),
	           const GhostTrackPrediction &prior = GhostTrackPrediction(),
	           const GlobalPoint &origin = GlobalPoint(),
	           bool withOrigin = false);

	const GhostTrackPrediction &prediction() const { return prediction_; }
	const GhostTrackPrediction &prior() const { return prior_; }

	const std::vector<GhostTrackState> &states() const { return states_; }
	double ndof() const { return ndof_; }
	double chi2() const { return chi2_; }

	operator Track() const { return prediction_.track(ndof_, chi2_); }

    private:
	void initStates(const std::vector<TransientTrack> &tracks,
	                const std::vector<float> &weights, double offset);

	GhostTrackPrediction		prediction_;
	GhostTrackPrediction		prior_;
	std::vector<GhostTrackState>	states_;
	double				ndof_;
	double				chi2_;
};

}

#endif // RecoBTag_GhostTrack_h

#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"

using namespace reco;

void GhostTrack::initStates(const std::vector<TransientTrack> &tracks,
                            const std::vector<float> &weights, double offset)
{
	std::vector<float>::const_iterator weight = weights.begin();
	for(std::vector<TransientTrack>::const_iterator iter = tracks.begin();
	    iter != tracks.end(); ++iter) {
		GhostTrackState state(*iter);
		state.linearize(prediction_, true, offset);
		if (weight != weights.end())
			state.setWeight(*weight++);

		states_.push_back(state);
	}
}

GhostTrack::GhostTrack(const GhostTrackPrediction &prior,
                       const GhostTrackPrediction &prediction,
                       const std::vector<TransientTrack> &tracks,
                       double ndof, double chi2,
                       const std::vector<float> &weights,
                       const GlobalPoint &origin,
                       bool withOrigin) :
	prediction_(prediction), prior_(prior),
	ndof_(ndof), chi2_(chi2)
{
	initStates(tracks, weights,
	           withOrigin ? prediction_.lambda(origin) : 0.);
}

GhostTrack::GhostTrack(const Track &ghostTrack,
                       const std::vector<TransientTrack> &tracks,
                       const std::vector<float> &weights,
                       const GhostTrackPrediction &prior,
                       const GlobalPoint &origin,
                       bool withOrigin) :
	prediction_(ghostTrack), prior_(prior),
	ndof_(ghostTrack.ndof()), chi2_(ghostTrack.chi2())
{
	initStates(tracks, weights,
	           withOrigin ? prediction_.lambda(origin) : 0.);
}

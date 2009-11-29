#ifndef RecoBTag_TrackGhostTrackState_h
#define RecoBTag_TrackGhostTrackState_h

#include <utility>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoVertex/GhostTrackFitter/interface/BasicGhostTrackState.h"

namespace reco {

class GhostTrackPrediction;

class TrackGhostTrackState : public BasicGhostTrackState {
    public:
	TrackGhostTrackState(const TransientTrack &track) : track_(track) {}

	const TransientTrack &track() const { return track_; }

	bool linearize(const GhostTrackPrediction &pred,
	               bool initial, double lambda);
	bool linearize(const GhostTrackPrediction &pred, double lambda);

	Vertex vertexStateOnGhostTrack(const GhostTrackPrediction &pred,
	                               bool withMeasurementError = true) const;
	Vertex vertexStateOnMeasurement(const GhostTrackPrediction &pred,
	                                bool withGhostTrackError = true) const;

    private:
	BasicGhostTrackState *clone() const;

	TransientTrack	track_;
};

}

#endif // RecoBTag_TrackGhostTrackState_h

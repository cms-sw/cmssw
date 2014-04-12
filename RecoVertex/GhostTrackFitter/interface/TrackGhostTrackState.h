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
	const TrajectoryStateOnSurface &tsos() const { return tsos_; }

	bool isValid() const { return tsos_.isValid(); }

	GlobalPoint globalPosition() const
	{ return tsos_.globalPosition(); }
	GlobalError cartesianError() const
	{ return tsos_.cartesianError().position(); }
	CovarianceMatrix cartesianCovariance() const
	{ return tsos_.cartesianError().matrix().Sub<CovarianceMatrix>(0, 0); }

	void reset() { tsos_ = TrajectoryStateOnSurface(); }
	bool linearize(const GhostTrackPrediction &pred,
	               bool initial, double lambda);
	bool linearize(const GhostTrackPrediction &pred, double lambda);

	Vertex vertexStateOnGhostTrack(const GhostTrackPrediction &pred,
	                               bool withMeasurementError) const;
	Vertex vertexStateOnMeasurement(const GhostTrackPrediction &pred,
	                                bool withGhostTrackError) const;

    private:
	BasicGhostTrackState *clone() const
	{ return new TrackGhostTrackState(*this); }

	TrajectoryStateOnSurface	tsos_;
	TransientTrack			track_;
};

}

#endif // RecoBTag_TrackGhostTrackState_h

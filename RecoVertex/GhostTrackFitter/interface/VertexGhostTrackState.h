#ifndef RecoBTag_VertexGhostTrackState_h
#define RecoBTag_VertexGhostTrackState_h

#include <utility>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "RecoVertex/GhostTrackFitter/interface/BasicGhostTrackState.h"

namespace reco {

class GhostTrackPrediction;

class VertexGhostTrackState : public BasicGhostTrackState {
    public:
	VertexGhostTrackState(const GlobalPoint &pos,
	                      const CovarianceMatrix &cov) :
		position_(pos), covariance_(cov) {}

	GlobalPoint globalPosition() const { return position_; }
	GlobalError cartesianError() const { return covariance_; }
	CovarianceMatrix cartesianCovariance() const { return covariance_; }

	Vertex vertexStateOnGhostTrack(const GhostTrackPrediction &pred,
	                               bool withMeasurementError) const;
	Vertex vertexStateOnMeasurement(const GhostTrackPrediction &pred,
	                                bool withGhostTrackError) const;

    private:
	BasicGhostTrackState *clone() const
	{ return new VertexGhostTrackState(*this); }

	GlobalPoint		position_;
	CovarianceMatrix	covariance_;
};

}

#endif // RecoBTag_VertexGhostTrackState_h

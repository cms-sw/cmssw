#ifndef RecoBTag_GhostTrackState_h
#define RecoBTag_GhostTrackState_h

#include <utility>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoVertex/GhostTrackFitter/interface/BasicGhostTrackState.h"

class VertexState;

namespace reco {

class TransientTrack;
class GhostTrackPrediction;

class GhostTrackState : public BasicGhostTrackState::Proxy {
	typedef BasicGhostTrackState::Proxy Base;

    public:
	typedef BasicGhostTrackState::CovarianceMatrix CovarianceMatrix;
	typedef BasicGhostTrackState::Vertex Vertex;

	GhostTrackState(const TransientTrack &track);
	GhostTrackState(const GlobalPoint &pos, const CovarianceMatrix &cov);
	GhostTrackState(const GlobalPoint &pos, const GlobalError &error);
	GhostTrackState(const VertexState &state);

	const TransientTrack &track() const;
	const TrajectoryStateOnSurface &tsos() const;

	GlobalPoint globalPosition() const { return data().globalPosition(); }
	GlobalError cartesianError() const { return data().cartesianError(); }
	CovarianceMatrix cartesianCovariance() const { return data().cartesianCovariance(); }

	double lambda() const { return data().lambda(); }
	double lambdaError(const GhostTrackPrediction &pred,
	                   const GlobalError &pvError = GlobalError()) const;
	bool isValid() const { return Base::isValid() && data().isValid(); }
	bool isTrack() const;
	bool isVertex() const;

	void reset() { unsharedData().reset(); }
	bool linearize(const GhostTrackPrediction &pred,
	               bool initial = false, double lambda = 0.)
	{ return unsharedData().linearize(pred, initial, lambda); }
	bool linearize(const GhostTrackPrediction &pred, double lambda)
	{ return unsharedData().linearize(pred, lambda); }

	double flightDistance(const GlobalPoint &point,
	                      const GlobalVector &dir) const;
	double axisDistance(const GlobalPoint &point,
	                    const GlobalVector &dir) const;
	double axisDistance(const GhostTrackPrediction &pred) const;

	Vertex vertexStateOnGhostTrack(
				const GhostTrackPrediction &pred,
				bool withMeasurementError = true) const
	{ return data().vertexStateOnGhostTrack(pred, withMeasurementError); }
	Vertex vertexStateOnMeasurement(const GhostTrackPrediction &pred,
	                                bool withGhostTrackError = true) const
	{ return data().vertexStateOnMeasurement(pred, withGhostTrackError); }

	double weight() const { return data().weight(); }
	void setWeight(double weight) { unsharedData().setWeight(weight); }
};

}

#endif // RecoBTag_GhostTrackState_h

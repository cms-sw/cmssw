#ifndef RecoBTag_BasicGhostTrackState_h
#define RecoBTag_BasicGhostTrackState_h

#include <utility>

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace reco {

class GhostTrackPrediction;

class BasicGhostTrackState : public ReferenceCountedInEvent {
    public:
	typedef BasicGhostTrackState			BGTS;
	typedef ProxyBase<BGTS, CopyUsingClone<BGTS> >	Proxy;

    private:
	friend class ProxyBase<BGTS, CopyUsingClone<BGTS> >;
	friend class ReferenceCountingPointer<BGTS>;
	friend class CopyUsingClone<BGTS>;

    public:
	typedef math::Error<3>::type CovarianceMatrix;
	typedef std::pair<GlobalPoint, GlobalError> Vertex;

	BasicGhostTrackState() : lambda_(0.), weight_(1.) {}
	~BasicGhostTrackState() override {}

	virtual GlobalPoint globalPosition() const = 0;
	virtual GlobalError cartesianError() const = 0;
	virtual CovarianceMatrix cartesianCovariance() const = 0;

	double lambda() const { return lambda_; }
	virtual bool isValid() const { return true; }

	virtual void reset() {}
	virtual bool linearize(const GhostTrackPrediction &pred,
	                       bool initial, double lambda)
	{ lambda_ = lambda; return true; }
	virtual bool linearize(const GhostTrackPrediction &pred,
	                       double lambda)
	{ lambda_ = lambda; return true; }

	virtual Vertex vertexStateOnGhostTrack(
				const GhostTrackPrediction &pred,
				bool withMeasurementError) const = 0;
	virtual Vertex vertexStateOnMeasurement(
				const GhostTrackPrediction &pred,
				bool withGhostTrackError) const = 0;

	double weight() const { return weight_; }
	void setWeight(double weight) { weight_ = weight; }

    protected:
	virtual BasicGhostTrackState *clone() const = 0;

	double	lambda_;
	double	weight_;
};

}

#endif // RecoBTag_BasicGhostTrackState_h

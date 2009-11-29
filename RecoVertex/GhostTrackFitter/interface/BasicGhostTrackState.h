#ifndef RecoBTag_BasicGhostTrackState_h
#define RecoBTag_BasicGhostTrackState_h

#include <utility>

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"

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
	typedef std::pair<GlobalPoint, GlobalError> Vertex;

	BasicGhostTrackState() : lambda_(0.), weight_(1.) {}
	virtual ~BasicGhostTrackState() {}

	const TrajectoryStateOnSurface &tsos() const { return tsos_; }

	double lambda() const { return lambda_; }
	bool isValid() const { return tsos_.isValid(); }

	void reset() { tsos_ = TrajectoryStateOnSurface(); }

	virtual bool linearize(const GhostTrackPrediction &pred,
	                       bool initial, double lambda) = 0;
	virtual bool linearize(const GhostTrackPrediction &pred,
	                       double lambda) = 0;

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

	TrajectoryStateOnSurface	tsos_;
	double				lambda_;
	double				weight_;
};

}

#endif // RecoBTag_BasicGhostTrackState_h

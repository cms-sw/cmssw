#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 

#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/TrackGhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/VertexGhostTrackState.h"

using namespace reco;

namespace {
	using namespace ROOT::Math;

	typedef SVector<double, 3> Vector3;

	inline Vector3 conv(const GlobalVector &vec)
	{
		Vector3 result;
		result[0] = vec.x();
		result[1] = vec.y();
		result[2] = vec.z();
		return result;
	}
}

GhostTrackState::GhostTrackState(const TransientTrack &track) :
	Base(new TrackGhostTrackState(track))
{
}

GhostTrackState::GhostTrackState(const GlobalPoint &pos,
                                 const CovarianceMatrix &cov) :
	Base(new VertexGhostTrackState(pos, cov))
{
}

GhostTrackState::GhostTrackState(const GlobalPoint &pos,
                                 const GlobalError &error) :
	Base(new VertexGhostTrackState(pos, error.matrix()))
{
}

GhostTrackState::GhostTrackState(const VertexState &state) :
	Base(new VertexGhostTrackState(state.position(),
	                               state.error().matrix()))
{
}

bool GhostTrackState::isTrack() const
{
	return dynamic_cast<const TrackGhostTrackState*>(&data()) != nullptr;
}

bool GhostTrackState::isVertex() const
{
	return dynamic_cast<const VertexGhostTrackState*>(&data()) != nullptr;
}

static const TrackGhostTrackState *getTrack(const BasicGhostTrackState *basic)
{
	const TrackGhostTrackState *track =
			dynamic_cast<const TrackGhostTrackState*>(basic);
	if (!track)
		throw cms::Exception("InvalidOperation")
			<< "track requested on non non-track GhostTrackState";
	return track;
}

const TransientTrack &GhostTrackState::track() const
{
	return getTrack(&data())->track();
}

const TrajectoryStateOnSurface &GhostTrackState::tsos() const
{
	return getTrack(&data())->tsos();
}

double GhostTrackState::flightDistance(const GlobalPoint &point,
                                       const GlobalVector &dir) const
{
	return (globalPosition() - point).dot(dir.unit());
}

double GhostTrackState::axisDistance(const GlobalPoint &point,
                                     const GlobalVector &dir) const
{
	return (globalPosition() - point).cross(dir.unit()).mag();
}

double GhostTrackState::axisDistance(const GhostTrackPrediction &pred) const
{
	return axisDistance(pred.origin(), pred.direction());
}

double GhostTrackState::lambdaError(const GhostTrackPrediction &pred,
                                    const GlobalError &pvError) const
{
	if (!isValid())
		return -1.;

	return std::sqrt(
	       	ROOT::Math::Similarity(
	       		conv(pred.direction()),
	       		(vertexStateOnGhostTrack(pred).second.matrix() +
			 pvError.matrix()))
	        / pred.rho2());
}

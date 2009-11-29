#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/TrackGhostTrackState.h"

using namespace reco;

namespace {
	using namespace ROOT::Math;

	typedef SVector<double, 3> Vector3;

	static inline Vector3 conv(const GlobalVector &vec)
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

bool GhostTrackState::isTrack() const
{
	return dynamic_cast<const TrackGhostTrackState*>(&data()) != 0;
}

bool GhostTrackState::isVertex() const
{
//	return dynamic_cast<const VertexGhostTrackState*>(&data()) != 0;
	return false;
}

const TransientTrack &GhostTrackState::track() const
{
	const TrackGhostTrackState *track =
			dynamic_cast<const TrackGhostTrackState*>(&data());

	if (!track)
		throw cms::Exception("InvalidOperation")
			<< "GhostTrackState::track() called non non-track";

	return track->track();
}

double GhostTrackState::flightDistance(const GlobalPoint &point,
                                       const GlobalVector &dir) const
{
	return (tsos().globalPosition() - point).dot(dir.unit());
}

double GhostTrackState::axisDistance(const GlobalPoint &point,
                                     const GlobalVector &dir) const
{
	return (tsos().globalPosition() - point).cross(dir.unit()).mag();
}

double GhostTrackState::axisDistance(const GhostTrackPrediction &pred) const
{
	return axisDistance(pred.origin(), pred.direction());
}

double GhostTrackState::lambdaError(const GhostTrackPrediction &pred,
                                    const GlobalError &pvError) const
{
	if (!tsos().isValid())
		return -1.;

	return std::sqrt(
	       	ROOT::Math::Similarity(
	       		conv(pred.direction()),
	       		(vertexStateOnGhostTrack(pred).second.matrix_new() +
			 pvError.matrix_new()))
	        / pred.rho2());
}

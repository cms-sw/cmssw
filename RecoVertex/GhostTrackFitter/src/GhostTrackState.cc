#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometrySurface/interface/Line.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"

using namespace reco;

bool GhostTrackState::linearize(const GhostTrackPrediction &pred,
                                bool initial, double lambda)
{
	AnalyticalTrajectoryExtrapolatorToLine extrap(track_.field());

	GlobalPoint origin = pred.origin();
	GlobalVector direction = pred.direction();

	if (tsos_.isValid() && !initial) {
		GlobalPoint pca = origin + lambda_ * direction;
		Line line(pca, direction);
		tsos_ = extrap.extrapolate(tsos_, line);
	} else {
		GlobalPoint pca = origin + lambda * direction;
		Line line(pca, direction);
		tsos_ = extrap.extrapolate(track_.impactPointState(), line);
	}

	if (!tsos_.isValid())
		return false;

	lambda_ = (tsos_.globalPosition() - origin) * direction / pred.rho2();

	return true;
}

bool GhostTrackState::linearize(const GhostTrackPrediction &pred,
                                double lambda)
{
	AnalyticalImpactPointExtrapolator extrap(track_.field());

	GlobalPoint point = pred.origin() + lambda * pred.direction();

	tsos_ = extrap.extrapolate(track_.impactPointState(), point);
	if (!tsos_.isValid())
		return false;

	lambda_ = lambda;

	return true;
}

double GhostTrackState::flightDistance(const GlobalPoint &point,
                                       const GlobalVector &dir) const
{
	return (tsos_.globalPosition() - point).dot(dir.unit());
}

double GhostTrackState::axisDistance(const GlobalPoint &point,
                                     const GlobalVector &dir) const
{
	return (tsos_.globalPosition() - point).cross(dir.unit()).mag();
}

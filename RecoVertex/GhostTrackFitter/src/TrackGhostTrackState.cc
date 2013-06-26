#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometrySurface/interface/Line.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/TrackGhostTrackState.h"

using namespace reco;

namespace {
	static inline double sqr(double arg) { return arg * arg; }

	using namespace ROOT::Math;

	typedef SVector<double, 3> Vector3;

	typedef SMatrix<double, 3, 3, MatRepSym<double, 3> > Matrix3S;
	typedef SMatrix<double, 6, 6, MatRepSym<double, 6> > Matrix6S;
	typedef SMatrix<double, 3, 3> Matrix33;
	typedef SMatrix<double, 3, 6> Matrix36;

	static inline Vector3 conv(const GlobalVector &vec)
	{
		Vector3 result;
		result[0] = vec.x();
		result[1] = vec.y();
		result[2] = vec.z();
		return result;
	}
}

bool TrackGhostTrackState::linearize(const GhostTrackPrediction &pred,
                                     bool initial, double lambda)
{
	AnalyticalTrajectoryExtrapolatorToLine extrap(track_.field());

	GlobalPoint origin = pred.origin();
	GlobalVector direction = pred.direction();

	if (isValid() && !initial) {
		GlobalPoint pca = origin + lambda_ * direction;
		Line line(pca, direction);
		tsos_ = extrap.extrapolate(tsos_, line);
	} else {
		GlobalPoint pca = origin + lambda * direction;
		Line line(pca, direction);
		tsos_ = extrap.extrapolate(track_.impactPointState(), line);
	}

	if (!isValid())
		return false;

	lambda_ = (tsos_.globalPosition() - origin) * direction / pred.rho2();

	return true;
}

bool TrackGhostTrackState::linearize(const GhostTrackPrediction &pred,
                                     double lambda)
{
	AnalyticalImpactPointExtrapolator extrap(track_.field());

	GlobalPoint point = pred.position(lambda);

	tsos_ = extrap.extrapolate(track_.impactPointState(), point);
	if (!isValid())
		return false;

	lambda_ = lambda;

	return true;
}

BasicGhostTrackState::Vertex TrackGhostTrackState::vertexStateOnGhostTrack(
	const GhostTrackPrediction &pred, bool withMeasurementError) const
{
	using namespace ROOT::Math;

	if (!isValid())
		return Vertex();

	GlobalPoint origin = pred.origin();
	GlobalVector direction = pred.direction();

	double rho2 = pred.rho2();
	double rho = std::sqrt(rho2);
	double lambda = (tsos_.globalPosition() - origin) * direction / rho2;
	GlobalPoint pos = origin + lambda * direction;

	GlobalVector momentum = tsos_.globalMomentum();
	double mom = momentum.mag();

	Vector3 b = conv(direction) / rho;
	Vector3 d = conv(momentum) / mom;
	double l = Dot(b, d);
	double g = 1. / (1. - sqr(l));

	Vector3 ca = conv(pos - tsos_.globalPosition());
	Vector3 bd = b - l * d;
	b *= g;

	Matrix33 pA = TensorProd(b, bd);
	Matrix33 pB = TensorProd(b, ca);

	Matrix36 jacobian;
	jacobian.Place_at(-pA + Matrix33(SMatrixIdentity()), 0, 0);
	jacobian.Place_at(pB / rho, 0, 3);
	Matrix3S error = Similarity(jacobian, pred.cartesianError(lambda));

	if (withMeasurementError) {
		jacobian.Place_at(pA, 0, 0);
		jacobian.Place_at(-pB / mom, 0, 3);
		error += Similarity(jacobian, tsos_.cartesianError().matrix());
	}

	return Vertex(pos, error);
}

BasicGhostTrackState::Vertex TrackGhostTrackState::vertexStateOnMeasurement(
	const GhostTrackPrediction &pred, bool withGhostTrackError) const
{
	using namespace ROOT::Math;

	if (!isValid())
		return Vertex();

	GlobalPoint origin = pred.origin();
	GlobalVector direction = pred.direction();

	double rho2 = pred.rho2();
	double rho = std::sqrt(rho2);
	double lambda = (tsos_.globalPosition() - origin) * direction / rho2;
	GlobalPoint pos = origin + lambda * direction;

	GlobalVector momentum = tsos_.globalMomentum();
	double mom = momentum.mag();

	Vector3 b = conv(direction) / rho;
	Vector3 d = conv(momentum) / mom;
	double l = Dot(b, d);
	double g = 1. / (1. - sqr(l));

	Vector3 ca = conv(tsos_.globalPosition() - pos);
	Vector3 bd = l * b - d;
	d *= g;

	Matrix33 pC = TensorProd(d, bd);
	Matrix33 pD = TensorProd(d, ca);

	Matrix36 jacobian;
	jacobian.Place_at(pC + Matrix33(SMatrixIdentity()), 0, 0);
	jacobian.Place_at(pD / mom, 0, 3);
	Matrix3S error = Similarity(jacobian, tsos_.cartesianError().matrix());

	if (withGhostTrackError) {
		jacobian.Place_at(-pC, 0, 0);
		jacobian.Place_at(-pD / rho, 0, 3);
		error += Similarity(jacobian, pred.cartesianError(lambda));
	}

	return Vertex(tsos_.globalPosition(), error);
}

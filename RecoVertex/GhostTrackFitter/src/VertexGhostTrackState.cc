#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/VertexGhostTrackState.h"

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

BasicGhostTrackState::Vertex VertexGhostTrackState::vertexStateOnGhostTrack(
	const GhostTrackPrediction &pred, bool withMeasurementError) const
{
	using namespace ROOT::Math;

	GlobalPoint origin = pred.origin();
	GlobalVector direction = pred.direction();

#if 0
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
#else	
	return Vertex();
#endif
}

BasicGhostTrackState::Vertex VertexGhostTrackState::vertexStateOnMeasurement(
	const GhostTrackPrediction &pred, bool withGhostTrackError) const
{
	return Vertex(position_, covariance_);
}

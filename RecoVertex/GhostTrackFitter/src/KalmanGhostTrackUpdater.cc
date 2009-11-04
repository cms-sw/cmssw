#include <vector>
#include <cmath>

#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"

#include "RecoVertex/GhostTrackFitter/interface/KalmanGhostTrackUpdater.h"

using namespace reco;

namespace {
	static inline double sqr(double arg) { return arg * arg; }

	using namespace ROOT::Math;

	typedef SVector<double, 4> Vector4;
	typedef SVector<double, 2> Vector2;

	typedef SMatrix<double, 4, 4, MatRepSym<double, 4> > Matrix4S;
	typedef SMatrix<double, 3, 3, MatRepSym<double, 3> > Matrix3S;
	typedef SMatrix<double, 2, 2, MatRepSym<double, 2> > Matrix2S;
	typedef SMatrix<double, 4, 4> Matrix44;
	typedef SMatrix<double, 4, 2> Matrix42;
	typedef SMatrix<double, 2, 4> Matrix24;
	typedef SMatrix<double, 2, 3> Matrix23;
	typedef SMatrix<double, 2, 2> Matrix22;

	struct KalmanState {
		KalmanState(const GhostTrackPrediction &pred,
		            const GhostTrackState &state);

		Vector2		residual;
		Matrix2S	measErr, measPredErr;
		Matrix24	h;
	};
}

KalmanState::KalmanState(const GhostTrackPrediction &pred,
                         const GhostTrackState &state)
{
	using namespace ROOT::Math;

	const GlobalPoint &point = state.tsos().globalPosition();

	// lambda
	GlobalVector dir = pred.direction();
	double rho2 = pred.rho2();
	double l = (point - pred.origin()) * dir / rho2;

	// jacobian of global -> local
	Matrix23 measToLocal;
	measToLocal(0, 2) = rho2;
	measToLocal(1, 0) = dir.y();
	measToLocal(1, 1) = -dir.x();

	// measurement in local 2d plane projection
	Vector2 meas(rho2 * point.z(),
	             dir.y() * point.x() - dir.x() * point.y());
	measErr = Similarity(measToLocal,
		state.tsos().cartesianError().matrix().Sub<Matrix3S>(0, 0));

	// jacobian of representation to measurement transformation
	h(0, 0) = 1.;
	h(0, 2) = l;
	h(1, 1) = -1.;
	h(1, 3) = -l;

	// predicted measurement
	Vector2 measPred(rho2 * (pred.z() + l * pred.cotTheta()), -pred.ip());
	measPredErr = Similarity(h, pred.covariance());

	// residual
	residual = meas - measPred;
}

GhostTrackPrediction KalmanGhostTrackUpdater::update(
					const GhostTrackPrediction &pred,
					const GhostTrackState &state,
					double &ndof, double &chi2) const
{
	using namespace ROOT::Math;

	KalmanState kalmanState(pred, state);

	if (state.weight() < 1.0e-3)
		return pred;

	// inverted combined error
	Matrix2S invErr = kalmanState.measPredErr +
	                  (1.0 / state.weight()) * kalmanState.measErr;
	if (!invErr.Invert())
		return pred;

	// gain
	Matrix42 gain = pred.covariance() * Transpose(kalmanState.h) * invErr;

	// new prediction
	Vector4 newPred = pred.prediction() + (gain * kalmanState.residual);
	Matrix44 tmp44 = SMatrixIdentity();
	tmp44 = (tmp44 - gain * kalmanState.h) * pred.covariance();
	Matrix4S newError(tmp44.LowerBlock());

	// filtered residuals
	Matrix22 tmp22 = SMatrixIdentity();
	tmp22 = (tmp22 - kalmanState.h * gain);
	Vector2 filtRes = tmp22 * kalmanState.residual;
	tmp22 *= kalmanState.measErr;
	Matrix2S filtResErr(tmp22.LowerBlock());
	if (!filtResErr.Invert())
		return pred;

	ndof += state.weight() * 2.;
	chi2 += state.weight() * Similarity(filtRes, filtResErr);

	return GhostTrackPrediction(newPred, newError);
}

void KalmanGhostTrackUpdater::contribution(
				const GhostTrackPrediction &pred,
				const GhostTrackState &state,
				double &ndof, double &chi2) const
{
	using namespace ROOT::Math;

	KalmanState kalmanState(pred, state);

	// this is called on the full predicted state,
	// so the residual is already with respect to the filtered state

	// inverted error
	Matrix2S invErr = kalmanState.measErr;
	if (!invErr.Invert()) {
		ndof = 0.;
		chi2 = 0.;
	}

	ndof = 2.;
	chi2 = Similarity(kalmanState.residual, invErr);
}                            

#include <memory>
#include <vector>
#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometryVector/interface/VectorUtil.h" 
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"

#include "RecoVertex/GhostTrackFitter/interface/AnnealingGhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/PositiveSideGhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/KalmanGhostTrackUpdater.h"

using namespace reco;

namespace {
	static inline double sqr(double arg) { return arg * arg; }

	using namespace ROOT::Math;

	typedef SVector<double, 4> Vector4;

	typedef SMatrix<double, 6, 6, MatRepSym<double, 6> > Matrix6S;
	typedef SMatrix<double, 4, 4, MatRepSym<double, 4> > Matrix4S;
	typedef SMatrix<double, 4, 6> Matrix46;
}

GhostTrackFitter::GhostTrackFitter()
{
	fitter.reset(new AnnealingGhostTrackFitter);
	updater.reset(new KalmanGhostTrackUpdater);
}

GhostTrackFitter::~GhostTrackFitter()
{
}

GhostTrack GhostTrackFitter::fit(
			const GlobalPoint &priorPosition,
			const GlobalError &priorError,
			const GlobalVector &direction,
			double coneRadius,
			const std::vector<TransientTrack> &tracks) const
{
	double dTheta = std::cosh((double)direction.eta()) * coneRadius;

	double r2 = direction.mag2();
	double r = std::sqrt(r2);
	double perp = direction.perp();
	double P = direction.x() / perp;
	double p = direction.y() / perp;
	double T = direction.z() / r;
	double t = perp / r;
	double h2 = dTheta * dTheta;
	double d2 = coneRadius * coneRadius;

	GlobalError cov(r2 * (T*T * P*P * h2 + t*t * p*p * d2),
	                r2 * p*P * (T*T * h2 - t*t * d2),
	                r2 * (T*T * p*p * h2 + t*t * P*P * d2),
	                -r2 * t*T * P * h2,
	                -r2 * t*T * p * h2,
	                r2 * t*t * h2);

	return fit(priorPosition, priorError, direction, cov, tracks);
}

GhostTrack GhostTrackFitter::fit(
			const GlobalPoint &priorPosition,
			const GlobalError &priorError,
			const GlobalVector &direction,
			const GlobalError &directionError,
			const std::vector<TransientTrack> &tracks) const
{
	using namespace ROOT::Math;

	double perp2 = direction.perp2();
	GlobalVector dir = direction / std::sqrt(perp2);
	double tip = priorPosition.y() * dir.x() - priorPosition.x() * dir.y();
	double l = priorPosition.x() * dir.x() + priorPosition.y() * dir.y();

	Vector4 vec;
	vec[0] = priorPosition.z() - dir.z() * l;
	vec[1] = tip;
	vec[2] = dir.z();
	vec[3] = dir.phi();

	Matrix46 jacobian;
	jacobian(0, 0) = -dir.x() * dir.z();
	jacobian(1, 0) = -dir.y();
	jacobian(0, 1) = -dir.y() * dir.z();
	jacobian(1, 1) = dir.x();
	jacobian(0, 2) = 1.;
	jacobian(0, 3) = -dir.z() * priorPosition.x();
	jacobian(1, 3) = priorPosition.y();
	jacobian(3, 3) = -dir.y();
	jacobian(0, 4) = -dir.z() * priorPosition.y();
	jacobian(1, 4) = -priorPosition.x();
	jacobian(3, 4) = dir.x();
	jacobian(0, 5) = -l;
	jacobian(2, 5) = 1.;

	Matrix6S origCov;
	origCov.Place_at(priorError.matrix_new(), 0, 0);
	origCov.Place_at(directionError.matrix_new() / perp2, 3, 3);

	Matrix4S cov = Similarity(jacobian, origCov);

	GhostTrackPrediction prior(vec, cov);

	double origin = (priorPosition - prior.origin()) * dir / prior.rho2();

	std::vector<GhostTrackState> states;
	for(std::vector<TransientTrack>::const_iterator iter = tracks.begin();
	    iter != tracks.end(); ++iter) {
		GhostTrackState state(*iter);
		state.linearize(prior, origin);
		state.linearize(prior);
		states.push_back(state);
	}

	PositiveSideGhostTrackFitter actualFitter(priorPosition, *fitter);
	return fit(actualFitter, prior, states);
}

GhostTrack GhostTrackFitter::fit(
			const GhostTrackPrediction &prior,
			const std::vector<TransientTrack> &tracks) const
{
	std::vector<GhostTrackState> states;
	for(std::vector<TransientTrack>::const_iterator iter = tracks.begin();
	    iter != tracks.end(); ++iter) {
		GhostTrackState state(*iter);
		state.linearize(prior);
		states.push_back(state);
	}

	return fit(*fitter, prior, states);
}

GhostTrack GhostTrackFitter::fit(
			FitterImpl &fitterImpl,
			const GhostTrackPrediction &prior,
			const std::vector<GhostTrackState> &states_) const
{
	std::vector<GhostTrackState> states = states_;

// std::cout << "prior = " << prior.prediction() << std::endl;

	double ndof, chi2;
	GhostTrackPrediction pred =
			fitterImpl.fit(*updater, prior, states, ndof, chi2);

	GhostTrack result(prior, pred, states, ndof, chi2);

	return result;
}

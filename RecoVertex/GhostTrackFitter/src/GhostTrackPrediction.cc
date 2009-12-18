#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

using namespace reco;

namespace {
	using namespace ROOT::Math;

	typedef SMatrix<double, 3, 4> Matrix34;
	typedef SMatrix<double, 4, 5> Matrix45;
	typedef SMatrix<double, 5, 4> Matrix54;
}

static GhostTrackPrediction::Vector convert(
			const CurvilinearTrajectoryParameters &trajectory)
{
	return GhostTrackPrediction::Vector(
			trajectory.yT() / std::cos(trajectory.lambda()),
			trajectory.xT(),
			std::tan(trajectory.lambda()),
			trajectory.phi());
}

static inline GhostTrackPrediction::Vector convert(
			const GlobalTrajectoryParameters &trajectory)
{
	return convert(CurvilinearTrajectoryParameters(
			trajectory.position(), trajectory.momentum(), 0));
}

static GhostTrackPrediction::Error convert(
				const GhostTrackPrediction::Vector &pred,
				const CurvilinearTrajectoryError &error)
{
	using namespace ROOT::Math;

	double rho2 = pred[2] * pred[2] + 1.;
	double rho = std::sqrt(rho2);

	Matrix45 jacobian;
	jacobian(0, 1) = pred[0] * pred[2];
	jacobian(0, 4) = rho;
	jacobian(1, 3) = 1.;
	jacobian(2, 1) = rho2;
	jacobian(3, 2) = 1.;

	return Similarity(jacobian, error.matrix());
}

GhostTrackPrediction::GhostTrackPrediction(
			const CurvilinearTrajectoryParameters &trajectory,
			const CurvilinearTrajectoryError &error) :
	prediction_(convert(trajectory)),
	covariance_(convert(prediction_, error))
{
}

GhostTrackPrediction::GhostTrackPrediction(
			const GlobalTrajectoryParameters &trajectory,
			const CurvilinearTrajectoryError &error) :
	prediction_(convert(trajectory)),
	covariance_(convert(prediction_, error))
{
}

GhostTrackPrediction::GhostTrackPrediction(const Track &track) :
	prediction_(convert(
		GlobalTrajectoryParameters(
			GlobalPoint(track.vx(), track.vy(), track.vz()),
			GlobalVector(track.px(), track.py(), track.pz()),
			0, 0))),
	covariance_(convert(prediction_, track.covariance()))
{
}

GlobalError GhostTrackPrediction::positionError(double lambda) const
{
	using namespace ROOT::Math;

	double x = std::cos(phi());
	double y = std::sin(phi());

	Matrix34 jacobian;
	jacobian(0, 1) = -y;
	jacobian(0, 3) = -y * lambda - x * ip();
	jacobian(1, 1) = x;
	jacobian(1, 3) = x * lambda - y * ip();
	jacobian(2, 0) = 1.;
	jacobian(2, 2) = lambda;

	return Similarity(jacobian, covariance());
}

CurvilinearTrajectoryParameters
GhostTrackPrediction::curvilinearTrajectory() const
{
	return CurvilinearTrajectoryParameters(0., std::atan(cotTheta()),
	                                       phi(), ip(), sz(), false);
}

GlobalTrajectoryParameters GhostTrackPrediction::globalTrajectory(
				const MagneticField *fieldProvider) const
{
	return GlobalTrajectoryParameters(origin(), direction(),
	                                  0, fieldProvider);
}

CurvilinearTrajectoryError GhostTrackPrediction::curvilinearError() const
{
	double rho2I = 1. / rho2();
	double rhoI = std::sqrt(rho2I);

	Matrix54 jacobian;
	jacobian(1, 2) = rho2I;
	jacobian(2, 3) = 1.;
	jacobian(3, 1) = 1.;
	jacobian(4, 0) = rhoI;
	jacobian(4, 2) = - z() * rhoI * cotTheta() * rho2I;

	AlgebraicSymMatrix55 err = Similarity(jacobian, covariance());
	err(0, 0) = 1.;

	return CurvilinearTrajectoryError(err);
}

FreeTrajectoryState GhostTrackPrediction::fts(
				const MagneticField *fieldProvider) const
{
	return FreeTrajectoryState(globalTrajectory(fieldProvider),
	                                            curvilinearError());
}

Track GhostTrackPrediction::track(double ndof, double chi2) const
{
	GlobalPoint origin = this->origin();
	GlobalVector dir = direction().unit();

	Track::Point point(origin.x(), origin.y(), origin.z());
	Track::Vector vector(dir.x(), dir.y(), dir.z());

	return Track(chi2, ndof, point, vector, 0, curvilinearError());
}

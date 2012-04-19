#ifndef RecoBTag_GhostTrackPrediction_h
#define RecoBTag_GhostTrackPrediction_h

#include <cmath>

#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class MagneticField;
class CurvilinearTrajectoryParameters;
class GlobalTrajectoryParameters;
class CurvilinearTrajectoryError;
class FreeTrajectoryState;

namespace reco {

class GhostTrackPrediction {
    public:
	// z0, tIP, dz/dr, phi

	typedef ROOT::Math::SVector<double, 4> Vector;
	typedef ROOT::Math::SMatrix<double, 4, 4,
			ROOT::Math::MatRepSym<double, 4> > Error;
	typedef ROOT::Math::SMatrix<double, 6, 6,
			ROOT::Math::MatRepSym<double, 6> > CartesianError;

	GhostTrackPrediction() {}
	GhostTrackPrediction(const Vector &prediction, const Error &error) :
		prediction_(prediction), covariance_(error)
	{}

	GhostTrackPrediction(const GlobalPoint &priorPosition, 
	                     const GlobalError &priorError,
	                     const GlobalVector &direction,
	                     double coneRadius);
	GhostTrackPrediction(const GlobalPoint &priorPosition, 
	                     const GlobalError &priorError,
	                     const GlobalVector &direction,
	                     const GlobalError &directionError)
	{ init(priorPosition, priorError, direction, directionError); }

	GhostTrackPrediction(const CurvilinearTrajectoryParameters &trajectory,
	                     const CurvilinearTrajectoryError &error);
	GhostTrackPrediction(const GlobalTrajectoryParameters &trajectory,
	                     const CurvilinearTrajectoryError &error);
	GhostTrackPrediction(const Track &track);

	double z() const { return prediction_[0]; }
	double ip() const { return prediction_[1]; }
	double cotTheta() const { return prediction_[2]; }
	double phi() const { return prediction_[3]; }

	double rho2() const { return cotTheta() * cotTheta() + 1.; }
	double rho() const { return std::sqrt(rho2()); }
	double sz() const { return z() / rho(); }
	double theta() const { return M_PI_2 - std::atan(cotTheta()); }
	double eta() const { return -std::log(rho() - cotTheta()); }

	const Vector &prediction() const { return prediction_; }
	const Error &covariance() const { return covariance_; }

	const GlobalPoint origin() const
	{ return GlobalPoint(-std::sin(phi()) * ip(), std::cos(phi()) * ip(), z()); }
	const GlobalVector direction() const
	{ return GlobalVector(std::cos(phi()), std::sin(phi()), cotTheta()); }

	double lambda(const GlobalPoint &point) const
	{ return (point - origin()) * direction() / rho2(); }

	GlobalPoint position(double lambda = 0.) const
	{ return origin() + lambda * direction(); }
	GlobalError positionError(double lambda = 0.) const;

	CartesianError cartesianError(double lambda = 0.) const;

	CurvilinearTrajectoryParameters curvilinearTrajectory() const;
	GlobalTrajectoryParameters globalTrajectory(
				const MagneticField *fieldProvider) const;
	CurvilinearTrajectoryError curvilinearError() const;

	FreeTrajectoryState fts(const MagneticField *fieldProvider) const;

	Track track(double ndof = 0., double chi2 = 0.) const;

    private:
	void init(const GlobalPoint &priorPosition, 
	          const GlobalError &priorError,
	          const GlobalVector &direction,
	          const GlobalError &directionError);

	Vector	prediction_;
	Error	covariance_;
};

}
#endif // RecoBTag_GhostTrackPrediction_h

#include <cmath>

#include <Math/SMatrix.h>
#include <Math/MatrixFunctions.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h" 
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

using namespace reco;

namespace {
	using namespace ROOT::Math;

	typedef SMatrix<double, 3, 4> Matrix34;
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

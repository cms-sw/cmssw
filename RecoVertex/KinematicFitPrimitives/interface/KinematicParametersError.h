#ifndef KinematicParametersError_H
#define KinematicParametersError_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

/**
 * Class to store the error matrix
 * for (x,y,z,p_x,p_y,p_z,m)
 * particle parametrization
 *
 * Kirill Prokofiev January 2003
 */

class KinematicParametersError {
public:
  KinematicParametersError() { vl = false; }

  KinematicParametersError(const AlgebraicSymMatrix77& er) : theCovMatrix(er) { vl = true; }

  KinematicParametersError(const CartesianTrajectoryError& err, float merr) {
    theCovMatrix.Place_at(err.matrix(), 0, 0);
    theCovMatrix(6, 6) = merr * merr;
    vl = true;
  }

  /**
 * access methods
 */

  AlgebraicSymMatrix77 const& matrix() const { return theCovMatrix; }

  AlgebraicSymMatrix77& matrix() { return theCovMatrix; }

  bool isValid() const { return vl; }

private:
  AlgebraicSymMatrix77 theCovMatrix;
  bool vl;
};
#endif

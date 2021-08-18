#ifndef PerigeeTrajectoryError_H
#define PerigeeTrajectoryError_H

#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "FWCore/Utilities/interface/Likely.h"
/**
 *  Class providing access to the <i> Perigee</i> parameters of a trajectory.
 *  These parameters consist of <BR>
 *  transverse curvature (signed), theta, phi,
 *  transverse impact parameter (signed), longitudinal i.p.
 */

class PerigeeTrajectoryError {
public:
  PerigeeTrajectoryError() {}
  // ~PerigeeTrajectoryError() {}

  /*
  PerigeeTrajectoryError(AlgebraicSymMatrix aPerigeeError):
    thePerigeeError(asSMatrix<5>(aPerigeeError)), weightIsAvailable(false) {}
  */
  PerigeeTrajectoryError(const AlgebraicSymMatrix55 &aPerigeeError)
      : thePerigeeError(aPerigeeError), weightIsAvailable(false) {}

  /**
   * The covariance matrix
   */
  const AlgebraicSymMatrix55 &covarianceMatrix() const { return thePerigeeError; }

  /**
   * The weight matrix (inverse of the covariance matrix)
   * The error variable is 0 in case of success.
   */
  const AlgebraicSymMatrix55 &weightMatrix(int &error) const {
    if UNLIKELY (!weightIsAvailable)
      calculateWeightMatrix();
    error = inverseError;
    return thePerigeeWeight;
  }

  void calculateWeightMatrix() const;

  double transverseCurvatureError() const { return sqrt(thePerigeeError(0, 0)); }

  /**
   * The theta angle
   */

  double thetaError() const { return sqrt(thePerigeeError(1, 1)); }

  /**
   * The phi angle
   */

  double phiError() const { return sqrt(thePerigeeError(2, 2)); }

  /**
   * The (signed) transverse impact parameter
   */

  double transverseImpactParameterError() const { return sqrt(thePerigeeError(3, 3)); }

  /**
   * The longitudinal impact parameter
   */

  double longitudinalImpactParameterError() const { return sqrt(thePerigeeError(4, 4)); }

private:
  AlgebraicSymMatrix55 thePerigeeError;
  mutable AlgebraicSymMatrix55 thePerigeeWeight;
  mutable int inverseError;
  mutable bool weightIsAvailable;
};
#endif

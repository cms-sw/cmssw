#ifndef _TRACKER_CURVILINEARTRAJECTORYERROR_H_
#define _TRACKER_CURVILINEARTRAJECTORYERROR_H_

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "DataFormats/Math/interface/Error.h"

/** Parametrization of the error matrix in the curvilinear frame.
 *  This frame is tangent to the track at the point of definition,
 *  with Z_T parallel to the track. X_T is in the global xy plane 
 *  and points to the left when looking into the direction of the track,
 *  and Y_T forms a right-handed frame with X_T and Z_T.
 * 
 *  The error along Z_T is therefore zero.
 *  The parameters are <BR>
 *    sigma^2( charge / abs_momentum) <BR>
 *    sigma^2( lambda) <BR>
 *    sigma^2( phi) <BR>
 *    sigma^2( x_transverse)) <BR>
 *    sigma^2( y_transverse)) <BR> <BR>
 *
 *  Please note that lambda and phi are defined in the global frame. Lambda is the helix
 *  dip angle (pi/2 minus theta (polar angle)), while phi is the angle of 
 *  inclination with the global x-axis in the transverse (global xy) plane.
 */

class CurvilinearTrajectoryError {
public:

  /// parameter dimension
  enum { dimension = 5 };
  /// 5 parameter covariance matrix
  typedef math::Error<dimension>::type MathCovarianceMatrix;

// construct
  CurvilinearTrajectoryError() {}

  /** Constructing class from a full covariance matrix. The sequence of the parameters is
   *  the same as the one described above.
   */

  CurvilinearTrajectoryError(const AlgebraicSymMatrix& aCovarianceMatrix) :
    theCovarianceMatrix(aCovarianceMatrix) {}

  /// Implicit conversion
  CurvilinearTrajectoryError( const MathCovarianceMatrix & cov);


// access

  /** Returning the covariance matrix.
   */

  AlgebraicSymMatrix matrix() const {
    return theCovarianceMatrix;
  }

  /** Enables the multiplication of the covariance matrix with the scalar "factor".
   */

  void operator *= (double factor) {
    theCovarianceMatrix *= factor;
  }

  operator MathCovarianceMatrix() const;

private:
  AlgebraicSymMatrix theCovarianceMatrix;
};

#endif

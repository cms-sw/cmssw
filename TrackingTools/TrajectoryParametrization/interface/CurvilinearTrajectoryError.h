#ifndef _TRACKER_CURVILINEARTRAJECTORYERROR_H_
#define _TRACKER_CURVILINEARTRAJECTORYERROR_H_

#include "DataFormats/GeometrySurface/interface/TrivialError.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
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

  CurvilinearTrajectoryError(InvalidError) : theCovarianceMatrix(ROOT::Math::SMatrixIdentity()) {theCovarianceMatrix(0,0)=-99999.e10;}

  /** Constructing class from a full covariance matrix. The sequence of the parameters is
   *  the same as the one described above.
   */
  CurvilinearTrajectoryError(const AlgebraicSymMatrix55& aCovarianceMatrix) :
    theCovarianceMatrix(aCovarianceMatrix) { }
  template<typename M55>
  CurvilinearTrajectoryError(const M55& aCovarianceMatrix) :
    theCovarianceMatrix(aCovarianceMatrix) { }


  bool invalid() const { return theCovarianceMatrix(0,0)<-1.e10;}
  bool valid() const { return !invalid();}

  // not really full check of posdef
  bool posDef() const { 
    return (theCovarianceMatrix(0,0)>=0) && (theCovarianceMatrix(1,1)>=0) && 
      (theCovarianceMatrix(2,2)>=0) && (theCovarianceMatrix(3,3)>=0) && (theCovarianceMatrix(4,4)>=0);
  }


// access

  /** Returning the covariance matrix.
   */
  const AlgebraicSymMatrix55 &matrix() const {
    return theCovarianceMatrix;
  }

 AlgebraicSymMatrix55 &matrix() {
    return theCovarianceMatrix;
  }


  /** Enables the multiplication of the covariance matrix with the scalar "factor".
   */

  void operator *= (double factor) {
    theCovarianceMatrix *= factor;
  }

  void zeroFieldScaling(double factor){
    double root_of_factor = sqrt(factor);
    //scale the 0 indexed covariance by the factor
    for (unsigned int i=1;i!=5;++i)      theCovarianceMatrix(i,0)*=root_of_factor;

    //scale all others by the scared factor
    for (unsigned int i=1;i!=5;++i)  for (unsigned int j=i;j!=5;++j) theCovarianceMatrix(i,j)*=factor;
    //term 0,0 is not scaled at all
  }

  operator MathCovarianceMatrix & () { return theCovarianceMatrix; }
  operator const MathCovarianceMatrix &() const { return theCovarianceMatrix; }

private:
  AlgebraicSymMatrix55 theCovarianceMatrix;
};

#endif

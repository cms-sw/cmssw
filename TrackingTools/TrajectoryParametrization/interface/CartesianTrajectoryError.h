#ifndef _TRACKER_CARTESIANTRAJECTORYERROR_H_
#define _TRACKER_CARTESIANTRAJECTORYERROR_H_

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

/** Class containing (6x6) error matrix of a track in the global, Cartesian frame.
 *  This error matrix should be used with care and in particular not as an
 *  intermediate frame when transforming between different, 5-dimensional
 *  track parameter frames. The order of the quantities inside the error matrix
 *  is the same as for the corresponding parameter vector provided by the
 *  GlobalTrajectoryParameters class.
 */

class CartesianTrajectoryError {
public:
// construct
  CartesianTrajectoryError() {}

  /** Constructing class from error matrix.
   */

  CartesianTrajectoryError(const AlgebraicSymMatrix66& aCovarianceMatrix) :
    theCovarianceMatrix(aCovarianceMatrix) { }

  /** Constructing class from error matrix.
   */

  CartesianTrajectoryError(const AlgebraicSymMatrix& aCovarianceMatrix) :
    theCovarianceMatrix(asSMatrix<6>(aCovarianceMatrix)) {}
// access

  /** Returning error matrix.
   */
  const AlgebraicSymMatrix matrix_old() const {
    return asHepMatrix(theCovarianceMatrix);
  }


  const AlgebraicSymMatrix66 &matrix() const {
    return theCovarianceMatrix;
  }

  /** Enables the multiplication of the error matrix with a scalar "factor".
   */

  void operator *= (double factor) {
    theCovarianceMatrix *= factor;
  }

  void zeroFieldScaling(double factor){
    double root_of_factor = sqrt(factor);
    //scale the 0 indexed covariance by the factor
    for (uint i=1;i!=5;++i)      theCovarianceMatrix(i,0)*=root_of_factor;

    //scale all others by the scared factor
    for (uint i=1;i!=5;++i)  for (uint j=i;j!=5;++j) theCovarianceMatrix(i,j)*=factor;
    //term 0,0 is not scaled at all
  }

  /// Position error submatrix

  /** Returning (3x3) submatrix of error matrix containing information about the errors
   *  and correlations between the different position coordinates. 
   */

  GlobalError position() const;

private:
  AlgebraicSymMatrix66 theCovarianceMatrix;
};

#endif

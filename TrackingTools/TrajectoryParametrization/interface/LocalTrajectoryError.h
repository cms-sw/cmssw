#ifndef _TRACKER_LOCALTRAJECTORYERROR_H_
#define _TRACKER_LOCALTRAJECTORYERROR_H_

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include <memory>

/** Class providing access to the covariance matrix of a set of relevant parameters of a trajectory
 *  in a local, Cartesian frame. The errors provided are: <BR> <BR>
 *  
 *  sigma^2(q/p) : charge (plus or minus one) divided by magnitude of momentum <BR>
 *  sigma^2(dxdz) : direction tangent in local xz-plane <BR>
 *  sigma^2(dydz) : direction tangent in local yz-plane <BR>
 *  sigma^2(x) : local x-coordinate <BR>
 *  sigma^2(y) : local y-coordinate <BR> <BR>
 *
 *  plus the relevant correlation terms.
 */

class LocalTrajectoryError {
public:
  // construct
  LocalTrajectoryError() {}

  LocalTrajectoryError(InvalidError) : theCovarianceMatrix(ROOT::Math::SMatrixNoInit()) {
    theCovarianceMatrix(0, 0) = -99999.e10;
  }
  // destruct
  ~LocalTrajectoryError() {}

  bool invalid() const { return theCovarianceMatrix(0, 0) < -1.e10; }
  bool valid() const { return !invalid(); }

  // not really full check of posdef
  bool posDef() const {
    return (theCovarianceMatrix(0, 0) >= 0) && (theCovarianceMatrix(1, 1) >= 0) && (theCovarianceMatrix(2, 2) >= 0) &&
           (theCovarianceMatrix(3, 3) >= 0) && (theCovarianceMatrix(4, 4) >= 0);
  }

  /** Constructing class from a full covariance matrix. The sequence of the parameters is
   *  the same as the one described above.
   */

  LocalTrajectoryError(const AlgebraicSymMatrix55 &aCovarianceMatrix)
      : theCovarianceMatrix(aCovarianceMatrix), theWeightMatrixPtr() {}

  /** Constructing class from standard deviations of the individual parameters, making
   *  the covariance matrix diagonal. The sequence of the input parameters is sigma(x), sigma(y),
   *  sigma(dxdz), sigma(dydz), sigma(q/p), but the resulting covariance matrix has the 
   *  same structure as the one described above.
   */

  LocalTrajectoryError(float dx, float dy, float dxdir, float dydir, float dpinv);

  // access

  /** Returns the covariance matrix.
   */

  const AlgebraicSymMatrix55 &matrix() const { return theCovarianceMatrix; }

  /** Returns the inverse of covariance matrix.
   */
  const AlgebraicSymMatrix55 &weightMatrix() const;

  /** Enables the multiplication of the covariance matrix with the scalar "factor".
   */

  LocalTrajectoryError &operator*=(double factor) {
    theCovarianceMatrix *= factor;
    if ((theWeightMatrixPtr.get() != nullptr) && (factor != 0.0)) {
      (*theWeightMatrixPtr) /= factor;
    }
    return *this;
  }

  /** Returns the two-by-two submatrix of the covariance matrix which yields the local
   *  position errors as well as the correlation between them.
   */

  LocalError positionError() const {
    return LocalError(theCovarianceMatrix(3, 3), theCovarianceMatrix(3, 4), theCovarianceMatrix(4, 4));
  }

private:
  AlgebraicSymMatrix55 theCovarianceMatrix;
  mutable std::shared_ptr<AlgebraicSymMatrix55> theWeightMatrixPtr;
};

#endif

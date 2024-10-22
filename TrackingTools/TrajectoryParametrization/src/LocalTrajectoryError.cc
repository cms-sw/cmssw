#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "FWCore/Utilities/interface/Likely.h"

LocalTrajectoryError::LocalTrajectoryError(float dx, float dy, float dxdir, float dydir, float dpinv)
    : theCovarianceMatrix(), theWeightMatrixPtr() {
  theCovarianceMatrix(3, 3) = dx * dx;
  theCovarianceMatrix(4, 4) = dy * dy;
  theCovarianceMatrix(1, 1) = dxdir * dxdir;
  theCovarianceMatrix(2, 2) = dydir * dydir;
  theCovarianceMatrix(0, 0) = dpinv * dpinv;
}

const AlgebraicSymMatrix55& LocalTrajectoryError::weightMatrix() const {
  if UNLIKELY (theWeightMatrixPtr.get() == nullptr) {
    theWeightMatrixPtr.reset(new AlgebraicSymMatrix55());
    invertPosDefMatrix(theCovarianceMatrix, *theWeightMatrixPtr);
  }
  return *theWeightMatrixPtr;
}

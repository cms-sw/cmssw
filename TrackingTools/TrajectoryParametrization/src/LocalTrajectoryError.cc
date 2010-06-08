#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

LocalTrajectoryError::LocalTrajectoryError() {}

LocalTrajectoryError::~LocalTrajectoryError() {}

LocalTrajectoryError::LocalTrajectoryError(const AlgebraicSymMatrix55& aCovarianceMatrix) :
  theCovarianceMatrix(aCovarianceMatrix), theWeightMatrixPtr() { }

LocalTrajectoryError::LocalTrajectoryError(const AlgebraicSymMatrix& aCovarianceMatrix) :
  theCovarianceMatrix(asSMatrix<5>(aCovarianceMatrix)), theWeightMatrixPtr() {}


LocalTrajectoryError::
LocalTrajectoryError( float dx, float dy, float dxdir, float dydir,
		      float dpinv) : theCovarianceMatrix(), theWeightMatrixPtr()
{
  theCovarianceMatrix(3,3) = dx*dx;
  theCovarianceMatrix(4,4) = dy*dy;
  theCovarianceMatrix(1,1) = dxdir*dxdir;
  theCovarianceMatrix(2,2) = dydir*dydir;
  theCovarianceMatrix(0,0) = dpinv*dpinv;
  
}

const AlgebraicSymMatrix55 & LocalTrajectoryError::weightMatrix() const {
  if (theWeightMatrixPtr.get() == 0) {
    theWeightMatrixPtr.reset(new AlgebraicSymMatrix55());
    invertPosDefMatrix(theCovarianceMatrix,*theWeightMatrixPtr);
  }
  return *theWeightMatrixPtr;
}

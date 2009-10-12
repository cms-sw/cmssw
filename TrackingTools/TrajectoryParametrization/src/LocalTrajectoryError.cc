#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"

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
/*LocalTrajectoryError::
LocalTrajectoryError( float dx, float dy, float dxdir, float dydir,
		      float dpinv)
{
  AlgebraicSymMatrix55 em;
  em(3,3) = dx*dx;
  em(4,4) = dy*dy;
  em(1,1) = dxdir*dxdir;
  em(2,2) = dydir*dydir;
  em(0,0) = dpinv*dpinv;
  
  theCovarianceMatrix = em;
}*/

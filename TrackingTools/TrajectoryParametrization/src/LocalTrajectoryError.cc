#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"

LocalTrajectoryError::
LocalTrajectoryError( float dx, float dy, float dxdir, float dydir,
		      float dpinv)
{
  AlgebraicSymMatrix em(5,0);
  em[3][3] = dx*dx;
  em[4][4] = dy*dy;
  em[1][1] = dxdir*dxdir;
  em[2][2] = dydir*dydir;
  em[0][0] = dpinv*dpinv;
  
  theCovarianceMatrix = em;
}

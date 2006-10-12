#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"

CurvilinearTrajectoryError::CurvilinearTrajectoryError( const MathCovarianceMatrix & cov) :
  theCovarianceMatrix(dimension)
{
  for (int i=0; i<dimension; i++) {
    for (int j=0; j<=i; j++) {
      theCovarianceMatrix(i+1, j+1) = cov(i,j);
    }
  }
}


CurvilinearTrajectoryError::operator MathCovarianceMatrix() const
{
  MathCovarianceMatrix cov;
  for (int i=0; i<dimension; i++) {
    for (int j=0; j<=i; j++) {
       cov(i,j) = theCovarianceMatrix(i+1, j+1);
    }
  }
  return cov;
}

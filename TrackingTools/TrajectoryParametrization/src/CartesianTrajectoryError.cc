#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"

GlobalError CartesianTrajectoryError::position() const {
  return GlobalError( theCovarianceMatrix.Sub<AlgebraicSymMatrix33>(0,0)); 
}

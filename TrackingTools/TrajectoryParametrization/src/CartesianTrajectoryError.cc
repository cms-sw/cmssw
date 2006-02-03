#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"

GlobalError CartesianTrajectoryError::position() const {
  return GlobalError( theCovarianceMatrix.sub(1,3));
}

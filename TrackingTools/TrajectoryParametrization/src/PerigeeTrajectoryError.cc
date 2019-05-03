#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"

void PerigeeTrajectoryError::calculateWeightMatrix() const {
  inverseError = invertPosDefMatrix(thePerigeeError, thePerigeeWeight) ? 0 : 1;
  weightIsAvailable = true;
}

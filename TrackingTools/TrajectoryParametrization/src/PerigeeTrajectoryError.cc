#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"


void calculateWeightMatrix() const {
  inverse error = invertPosDefMatrix(thePerigeeError, thePerigeeWeight) ?
    0 : 1;
  weightIsAvailable = true;
}

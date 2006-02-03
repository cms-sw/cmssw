#ifndef PerigeeTrajectoryError_H
#define PerigeeTrajectoryError_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"
/**
 *  Class providing access to the <i> Perigee</i> parameters of a trajectory.
 *  These parameters consist of <BR>
 *  transverse curvature (signed), theta, phi,
 *  transverse impact parameter (signed), longitudinal i.p.
 */

class PerigeeTrajectoryError
{

public:

  PerigeeTrajectoryError() {}

  PerigeeTrajectoryError(AlgebraicSymMatrix aPerigeeError):
    thePerigeeError(aPerigeeError), weightIsAvailable(false) {}

  /**
   * The covariance matrix
   */

  const AlgebraicSymMatrix & covarianceMatrix() const {return thePerigeeError;}

  /**
   * The weight matrix (inverse of the covariance matrix)
   */

  const AlgebraicSymMatrix & weightMatrix() const
  {
    if (!weightIsAvailable) {
      int error;
      thePerigeeWeight = thePerigeeError.inverse(error);
      if (error != 0 ) throw TrajectoryStateException(
	"PerigeeTrajectoryError::Unable to inverse covariance matrix"); 
      weightIsAvailable = true;
    }
    return thePerigeeWeight;
  }

private:
  AlgebraicSymMatrix thePerigeeError;
  mutable AlgebraicSymMatrix thePerigeeWeight;
  mutable bool weightIsAvailable;

};
#endif

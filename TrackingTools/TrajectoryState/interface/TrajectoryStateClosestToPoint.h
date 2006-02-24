#ifndef TrajectoryStateClosestToPoint_H
#define TrajectoryStateClosestToPoint_H

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

/**
 * Trajectory state defined at a given point on the helix, which is 
 * the point of closest approach to the reference point.
 * In addition to the FreeTrajectoryState at that point, it also 
 * gives the perigee parameters.
 */

class TrajectoryStateClosestToPoint
{
  typedef TrajectoryStateOnSurface	TSOS;
  typedef FreeTrajectoryState		FTS;

public:

  /**
   * returns the state defined at the point of closest approach to the
   * reference point.
   */

  const FreeTrajectoryState & theState() const {
    if (!theFTSavailable) throw TrajectoryStateException(
      "TrajectoryStateClosestToPoint: attempt to access FTS when none available");
    return theFTS;
  }


  /**
   * returns the reference point which used to construct the state.
   * It is thus the point with respect to which the impact parameters
   * are defined.
   */ 

  const GlobalPoint referencePoint() const {
    return theRefPoint;
  }


  /**
   * returns the perigee parameters at the p.c.a. to the reference 
   *  point.
   */

  const PerigeeTrajectoryParameters & perigeeParameters() const {
    return theParameters;
  }
  

  /**
   * returns the error of the perigee parameters if it is 
   * available
   */

  const PerigeeTrajectoryError & perigeeError() const {
    if (!errorIsAvailable) throw TrajectoryStateException(
      "TrajectoryStateClosestToPoint: attempt to access errors when none available");
    return thePerigeeError;
  }


  /**
   * tells whether the error of the perigee parameters 
   * is available.
   */

  bool hasError() const {
    return errorIsAvailable;
  }


public:

  TrajectoryStateClosestToPoint() {}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * no error is provided.
   */

  TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
    const GlobalPoint& referencePoint, const MagneticField* magField);

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * an error is provided.
   */

  TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
    const PerigeeTrajectoryError& perigeeError,
    const GlobalPoint& referencePoint, const MagneticField* magField);

private:

  friend class TrajectoryStateClosestToPointBuilder;
  friend class PerigeeConversions;

  /**
   * Use the appropriate TrajectoryStateClosestToPointBuilder to
   * get access to this constructor
   */
  
  TrajectoryStateClosestToPoint(const FTS& originalFTS, 
    const GlobalPoint& referencePoint);

  /**
   * Jacobians of tranformations between curvilinear frame at point of closest
   * approach in transverse plane and perigee frame. The fts must therefore be
   * given at exactly this point in order to yield the correct Jacobians.
   */

  AlgebraicMatrix jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts) const;
  AlgebraicMatrix jacobianPerigee2Curvilinear(const FreeTrajectoryState& fts) const;


  FTS theFTS;
  bool theFTSavailable;
  
  GlobalPoint theRefPoint;
  PerigeeTrajectoryParameters theParameters;
  PerigeeTrajectoryError thePerigeeError;
  bool errorIsAvailable;
  PerigeeConversions perigeeConversions;
  
};
#endif

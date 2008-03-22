#ifndef TrajectoryStateClosestToPoint_H
#define TrajectoryStateClosestToPoint_H

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"

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
  /// parameter dimension

public:

  TrajectoryStateClosestToPoint():
    theFTSavailable(false), errorIsAvailable(false) {}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * no error is provided.
   */

  TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters, double pt,
				const GlobalPoint& referencePoint, const MagneticField* field);

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * an error is provided.
   */

  TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters, double pt,
    const PerigeeTrajectoryError& perigeeError, const GlobalPoint& referencePoint,
    const MagneticField* field);


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
   * returns the transverse momentum magnitude
   */

  double pt() const { return thePt; }

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
   * returns the state defined at the point of closest approach to the
   * reference point.
   */

  GlobalPoint position() const {
    return perigeeConversions.positionFromPerigee(theParameters, theRefPoint);
  }


  GlobalVector momentum() const {
    return perigeeConversions.momentumFromPerigee(theParameters, thePt, theRefPoint);
  }


  TrackCharge charge() const {
    return theParameters.charge();
  }


  const FreeTrajectoryState & theState() const {
    if (!theFTSavailable) calculateFTS();
    return theFTS;
  }


  /**
   * tells whether the error of the perigee parameters 
   * is available.
   */

  bool hasError() const {
    return errorIsAvailable;
  }


private:

  friend class TrajectoryStateClosestToPointBuilder;
  friend class PerigeeConversions;

  /**
   * Use the appropriate TrajectoryStateClosestToPointBuilder to
   * get access to this constructor
   */
  
  TrajectoryStateClosestToPoint(const FTS& originalFTS, const GlobalPoint& referencePoint);

  void calculateFTS() const;

  const MagneticField* theField;

  mutable FTS theFTS;
  mutable bool theFTSavailable;
  
  GlobalPoint theRefPoint;
  PerigeeTrajectoryParameters theParameters;
  double thePt;
  PerigeeTrajectoryError thePerigeeError;
  bool errorIsAvailable;
  PerigeeConversions perigeeConversions;
  
};
#endif

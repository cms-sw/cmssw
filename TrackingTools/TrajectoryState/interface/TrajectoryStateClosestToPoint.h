#ifndef TrajectoryStateClosestToPoint_H
#define TrajectoryStateClosestToPoint_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

/**
 * Trajectory state defined at a given point on the helix, which is 
 * the point of closest approach to the reference point.
 * In addition to the FreeTrajectoryState at that point, it also 
 * gives the perigee parameters.
 * This state can also be invalid, e.g. in case the propagation was not successful.
 */

class TrajectoryStateClosestToPoint {
  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  /// parameter dimension

public:
  TrajectoryStateClosestToPoint() : valid(false), theFTSavailable(false), errorIsAvailable(false) {}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * no error is provided.
   */
  TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
                                double pt,
                                const GlobalPoint& referencePoint,
                                const MagneticField* field)
      : theField(field),
        theRefPoint(referencePoint),
        theParameters(perigeeParameters),
        thePt(pt),
        valid(true),
        theFTSavailable(false),
        errorIsAvailable(false) {}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * an error is provided.
   */
  TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
                                double pt,
                                const PerigeeTrajectoryError& perigeeError,
                                const GlobalPoint& referencePoint,
                                const MagneticField* field)
      : theField(field),
        theRefPoint(referencePoint),
        theParameters(perigeeParameters),
        thePt(pt),
        thePerigeeError(perigeeError),
        valid(true),
        theFTSavailable(false),
        errorIsAvailable(true) {}

  /**
   * returns the reference point which used to construct the state.
   * It is thus the point with respect to which the impact parameters
   * are defined.
   */
  const GlobalPoint& referencePoint() const { return theRefPoint; }

  /**
   * returns the perigee parameters at the p.c.a. to the reference 
   *  point.
   */
  const PerigeeTrajectoryParameters& perigeeParameters() const { return theParameters; }

  /**
   * returns the transverse momentum magnitude
   */
  double pt() const { return thePt; }

  /**
   * returns the error of the perigee parameters if it is 
   * available
   */
  const PerigeeTrajectoryError& perigeeError() const { return thePerigeeError; }

  /**
   * returns the state defined at the point of closest approach to the
   * reference point.
   */
  GlobalPoint position() const { return PerigeeConversions::positionFromPerigee(theParameters, theRefPoint); }

  GlobalVector momentum() const { return PerigeeConversions::momentumFromPerigee(theParameters, thePt, theRefPoint); }

  TrackCharge charge() const { return theParameters.charge(); }

  const FreeTrajectoryState& theState() const {
    if (!theFTSavailable)
      calculateFTS();
    return theFTS;
  }

  /**
   * tells whether the error of the perigee parameters 
   * is available.
   */
  bool hasError() const { return errorIsAvailable; }

  /**
   * Tells whether the state is valid or not
   */
  bool isValid() const { return valid; }

  friend class TrajectoryStateClosestToPointBuilder;

  /**
   * Use the appropriate TrajectoryStateClosestToPointBuilder to
   * get access to this constructor
   */

  TrajectoryStateClosestToPoint(const FTS& originalFTS, const GlobalPoint& referencePoint);

  void calculateFTS() const;

private:
  const MagneticField* theField;

  mutable FTS theFTS;

  GlobalPoint theRefPoint;
  PerigeeTrajectoryParameters theParameters;
  double thePt;
  PerigeeTrajectoryError thePerigeeError;
  bool valid;
  mutable bool theFTSavailable;
  bool errorIsAvailable;
};
#endif

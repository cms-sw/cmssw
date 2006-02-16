#ifndef _TRACKER_TRAJECTORYMEASUREMENT_H_
#define _TRACKER_TRAJECTORYMEASUREMENT_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

/** The TrajectoryMeasurement contains the full information about the
 *  measurement of a trajectory by a Det, namely <BR>
 *    - the TrackingRecHit <BR>
 *    - the predicted TrajectoryStateOnSurface from forward propagation (fitter)<BR>
 *    - the predicted TrajectoryStateOnSurface from backward propagation (smoother)<BR>
 *    - the (combination of the) predicted TrajectoryStateOnSurfaces updated with the TrackingRecHit information <BR>
 *    - the compatibility estimate between the TrackingRecHit and the predicted state. <BR>
 *
 *  A container of TrajectoryMeasurements is the result of querying a Det for
 *  measurements compatible with a TrajectoryState.
 *  A reconstructed track also consists of an ordered collection of 
 *  TrajectoryMeasurements.
 */

class TrajectoryMeasurement {
public:

  TrajectoryMeasurement() {}

  /// Constructor with forward predicted state, const TrackingRecHit*
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdTrajectoryStateOnSurface,
                        const TrackingRecHit* aRecHit) :
    theFwdPredictedState(fwdTrajectoryStateOnSurface),
    theUpdatedState(fwdTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theEstimate(0) {}

  /// Constructor with forward predicted state, RecHit, estimate
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdTrajectoryStateOnSurface,
                        const TrackingRecHit* aRecHit, float aEstimate) :
    theFwdPredictedState(fwdTrajectoryStateOnSurface),
    theUpdatedState(fwdTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theEstimate(aEstimate) {}

  /// Constructor with forward predicted & updated state, RecHit
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        const TrackingRecHit* aRecHit) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theEstimate(0) {}

  /// Constructor with forward predicted & updated state, RecHit, estimate 
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        const TrackingRecHit* aRecHit, float aEstimate) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theEstimate(aEstimate) {}

  /** Constructor with forward predicted, backward predicted & updated state, 
   *  RecHit
   */
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
			TrajectoryStateOnSurface bwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        const TrackingRecHit* aRecHit) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theBwdPredictedState(bwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theEstimate(0) {}

  /** Constructor with forward predicted, backward predicted & updated state, 
   *  RecHit, estimate
   */
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
			TrajectoryStateOnSurface bwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        const TrackingRecHit* aRecHit, float aEstimate) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theBwdPredictedState(bwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theEstimate(aEstimate) {}

  /** Access to forward predicted state (from fitter or builder).
   *  To be replaced by forwardPredictedState.
   */
  TrajectoryStateOnSurface predictedState() const {
    return theFwdPredictedState;
  }

  /// Access to forward predicted state (from fitter or builder)
  TrajectoryStateOnSurface forwardPredictedState() const {
    return theFwdPredictedState;
  }
  /// Access to backward predicted state (from smoother)
  TrajectoryStateOnSurface backwardPredictedState() const {
    return theBwdPredictedState;
  }

  /** Access to updated state (combination of forward predicted state
   *  and hit for fitter, + backward predicted state for smoother)
   */
  TrajectoryStateOnSurface updatedState() const {
    return theUpdatedState;
  }

  const TrackingRecHit* recHit() const {
    return theRecHit;
  }

  float estimate() const { return theEstimate;}

private:
  TrajectoryStateOnSurface theFwdPredictedState;
  TrajectoryStateOnSurface theBwdPredictedState;
  TrajectoryStateOnSurface theUpdatedState;
  const TrackingRecHit* theRecHit;
  float theEstimate;
};

#endif

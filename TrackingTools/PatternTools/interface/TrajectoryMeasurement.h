#ifndef _TRACKER_TRAJECTORYMEASUREMENT_H_
#define _TRACKER_TRAJECTORYMEASUREMENT_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "boost/intrusive_ptr.hpp" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include<algorithm>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class DetLayer;

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

  typedef TransientTrackingRecHit::RecHitPointer         RecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitPointer    ConstRecHitPointer;

  TrajectoryMeasurement() {}

  /// Constructor with forward predicted state, const TrackingRecHit*
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit) :
    theFwdPredictedState(fwdTrajectoryStateOnSurface),
    theUpdatedState(fwdTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theLayer(0), theEstimate(0) {}

  /// Constructor with forward predicted state, RecHit, estimate
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit, float aEstimate) :
    theFwdPredictedState(fwdTrajectoryStateOnSurface),
    theUpdatedState(fwdTrajectoryStateOnSurface),
    theRecHit(aRecHit), 
    theLayer(0),
    theEstimate(aEstimate) {}
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit, float aEstimate,
			const DetLayer* layer) :
    theFwdPredictedState(fwdTrajectoryStateOnSurface),
    theUpdatedState(fwdTrajectoryStateOnSurface),
    theRecHit(aRecHit), theLayer(layer) ,
    theEstimate(aEstimate){}

  /// Constructor with forward predicted & updated state, RecHit
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit), theLayer(0),
    theEstimate(0) {}

  /// Constructor with forward predicted & updated state, RecHit, estimate 
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit, float aEstimate) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit), theLayer(0),
    theEstimate(aEstimate) {}
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit, float aEstimate,
			const DetLayer* layer) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit), theLayer(layer),
    theEstimate(aEstimate) {}

  /** Constructor with forward predicted, backward predicted & updated state, 
   *  RecHit
   */
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
			TrajectoryStateOnSurface bwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theBwdPredictedState(bwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theLayer(0), theEstimate(0) {}

  /** Constructor with forward predicted, backward predicted & updated state, 
   *  RecHit, estimate
   */
  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
			TrajectoryStateOnSurface bwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit, float aEstimate) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theBwdPredictedState(bwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theLayer(0), theEstimate(aEstimate) {}

  TrajectoryMeasurement(TrajectoryStateOnSurface fwdPredTrajectoryStateOnSurface,
			TrajectoryStateOnSurface bwdPredTrajectoryStateOnSurface,
                        TrajectoryStateOnSurface uTrajectoryStateOnSurface,
                        ConstRecHitPointer aRecHit, float aEstimate,
			const DetLayer* layer) :
    theFwdPredictedState(fwdPredTrajectoryStateOnSurface),
    theBwdPredictedState(bwdPredTrajectoryStateOnSurface),
    theUpdatedState(uTrajectoryStateOnSurface),
    theRecHit(aRecHit),
    theLayer(layer), theEstimate(aEstimate) {}

 
  TrajectoryMeasurement( TrajectoryMeasurement const & rh) :
    theFwdPredictedState(rh.theFwdPredictedState),
    theBwdPredictedState(rh.theBwdPredictedState),
    theUpdatedState(rh.theUpdatedState),
    theRecHit(rh.theRecHit), theLayer(rh.theLayer),
    theEstimate(rh.theEstimate) {}
                        
  TrajectoryMeasurement & operator=( TrajectoryMeasurement const & rh) {   
    theFwdPredictedState = rh.theFwdPredictedState;   
    theBwdPredictedState = rh.theBwdPredictedState;
    theUpdatedState = rh.theUpdatedState;
    theRecHit = rh.theRecHit;
    theEstimate=rh.theEstimate;
    theLayer=rh.theLayer;

    return *this;

  }

#if defined( __GXX_EXPERIMENTAL_CXX0X__)

  TrajectoryMeasurement( TrajectoryMeasurement && rh)  noexcept:
    theFwdPredictedState(std::move(rh.theFwdPredictedState)),
    theBwdPredictedState(std::move(rh.theBwdPredictedState)),
    theUpdatedState(std::move(rh.theUpdatedState)),
    theRecHit(std::move(rh.theRecHit)), theLayer(rh.theLayer),
    theEstimate(rh.theEstimate) {}

  TrajectoryMeasurement & operator=( TrajectoryMeasurement && rh)  noexcept{
    using std::swap;
    swap(theFwdPredictedState,rh.theFwdPredictedState);
    swap(theBwdPredictedState,rh.theBwdPredictedState);
    swap(theUpdatedState,rh.theUpdatedState);
    swap(theRecHit,rh.theRecHit);
    theEstimate=rh.theEstimate;
    theLayer=rh.theLayer;

    return *this;

  }
  
#endif

  /** Access to forward predicted state (from fitter or builder).
   *  To be replaced by forwardPredictedState.
   */
  TrajectoryStateOnSurface const & predictedState() const {
    return theFwdPredictedState;
  }

  /// Access to forward predicted state (from fitter or builder)
  TrajectoryStateOnSurface const & forwardPredictedState() const {
    return theFwdPredictedState;
  }
  /// Access to backward predicted state (from smoother)
  TrajectoryStateOnSurface const & backwardPredictedState() const {
    return theBwdPredictedState;
  }

  /** Access to updated state (combination of forward predicted state
   *  and hit for fitter, + backward predicted state for smoother)
   */
  TrajectoryStateOnSurface const & updatedState() const {
    return theUpdatedState;
  }

  ConstRecHitPointer::element_type const & recHitR() const {
    return *theRecHit.get();
  }

  ConstRecHitPointer const & recHitP() const {
    return theRecHit;
  }

  ConstRecHitPointer const & recHit() const {
    return recHitP();
  } 

  float estimate() const { return theEstimate;}

  const DetLayer* layer() const { return theLayer;}

  void setLayer( DetLayer const * il) const { theLayer=il;}

private:
  TrajectoryStateOnSurface theFwdPredictedState;
  TrajectoryStateOnSurface theBwdPredictedState;
  TrajectoryStateOnSurface theUpdatedState;
  ConstRecHitPointer       theRecHit;
  mutable DetLayer const * theLayer;
  float theEstimate;
};

#endif

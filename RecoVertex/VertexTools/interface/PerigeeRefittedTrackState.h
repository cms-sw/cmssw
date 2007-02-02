#ifndef PerigeeRefittedTrackState_H
#define PerigeeRefittedTrackState_H

#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

/**
 * Caching refitted state of the trajectory after the vertex fit is
 * done. For the Perigee parametrization.
 */

class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class Surface;
class Propagator;

class PerigeeRefittedTrackState : public RefittedTrackState {

public:

  PerigeeRefittedTrackState(const TrajectoryStateClosestToPoint & tscp, 
  			    double aWeight = 1.) :
    theState(tscp), momentumVectorAvailable(false), theWeight(aWeight) {}

 virtual ~PerigeeRefittedTrackState(){}

  /**
   * Transformation into a FreeTrajectoryState
   */

  virtual FreeTrajectoryState freeTrajectoryState() const
    {return theState.theState();}

  /**
   * Transformation into a TSOS at a given surface
   */
  virtual TrajectoryStateOnSurface trajectoryStateOnSurface(
  		const Surface & surface) const;

  /**
   * Transformation into a TSOS at a given surface, with a given propagator
   */

  virtual TrajectoryStateOnSurface trajectoryStateOnSurface(
		const Surface & surface, const Propagator & propagator) const;

  /**
   * Vector containing the refitted track parameters. <br>
   * These are (signed transverse curvature, theta, phi,
   *  (signed) transverse , longitudinal impact parameter)
   */

  virtual AlgebraicVector  parameters() const
    {return theState.perigeeParameters().vector();}

  /**
   * The covariance matrix
   */

  virtual AlgebraicSymMatrix  covariance() const
    {return theState.perigeeError().covarianceMatrix();}

  /**
   * Position at which the momentum is defined.
   */

  virtual GlobalPoint position() const
    {return theState.referencePoint();}

  /**
   * Vector containing the parameters describing the momentum as the vertex.
   * These are (signed transverse curvature, theta, phi)
   */

  virtual AlgebraicVector momentumVector() const;

  /**
   *   The weight of this component in a mixture
   */
  virtual double weight() const {return theWeight;}

  /**
   * Returns a new refitted state of the same type, but with another weight.
   * The current state is unchanged.
   */
  virtual ReferenceCountingPointer<RefittedTrackState> stateWithNewWeight
  	(const double newWeight) const;

  virtual std::vector<ReferenceCountingPointer<RefittedTrackState> > components() const;

  virtual reco::TransientTrack transientTrack() const;

private:

  TrajectoryStateClosestToPoint theState;
  mutable bool momentumVectorAvailable;
  mutable AlgebraicVector momentumAtVertex;
  double theWeight;
};
#endif

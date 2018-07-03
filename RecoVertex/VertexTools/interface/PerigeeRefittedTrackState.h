#ifndef PerigeeRefittedTrackState_H
#define PerigeeRefittedTrackState_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
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

class PerigeeRefittedTrackState : public RefittedTrackState<5> {

public:

  typedef ReferenceCountingPointer<RefittedTrackState<5> > RefCountedRefittedTrackState;

  PerigeeRefittedTrackState(const TrajectoryStateClosestToPoint & tscp,
  			    const AlgebraicVector3 & aMomentumAtVertex,
  			    const double aWeight = 1.) :
    theState(tscp), momentumAtVertex(aMomentumAtVertex), theWeight(aWeight) {}

 ~PerigeeRefittedTrackState() override{}

  /**
   * Transformation into a FreeTrajectoryState
   */

  FreeTrajectoryState freeTrajectoryState() const override
    {return theState.theState();}

  /**
   * Transformation into a TSOS at a given surface
   */
  TrajectoryStateOnSurface trajectoryStateOnSurface(
  		const Surface & surface) const override;

  /**
   * Transformation into a TSOS at a given surface, with a given propagator
   */

  TrajectoryStateOnSurface trajectoryStateOnSurface(
		const Surface & surface, const Propagator & propagator) const override;

  /**
   * Vector containing the refitted track parameters. <br>
   * These are (signed transverse curvature, theta, phi,
   *  (signed) transverse , longitudinal impact parameter)
   */

  AlgebraicVector5 parameters() const override
    {return theState.perigeeParameters().vector();}

  /**
   * The covariance matrix
   */

  AlgebraicSymMatrix55  covariance() const override
    {return theState.perigeeError().covarianceMatrix();}

  /**
   * Position at which the momentum is defined.
   */

  GlobalPoint position() const override
    {return theState.referencePoint();}

  /**
   * Vector containing the parameters describing the momentum as the vertex.
   * These are (signed transverse curvature, theta, phi)
   */

  AlgebraicVector3 momentumVector() const override;

  /**
   *   The weight of this component in a mixture
   */
  double weight() const override {return theWeight;}

  /**
   * Returns a new refitted state of the same type, but with another weight.
   * The current state is unchanged.
   */
  ReferenceCountingPointer<RefittedTrackState<5> > stateWithNewWeight
  	(const double newWeight) const override;

  std::vector<ReferenceCountingPointer<RefittedTrackState<5> > > components() const override;

  reco::TransientTrack transientTrack() const override;

private:

  TrajectoryStateClosestToPoint theState;
  AlgebraicVector3 momentumAtVertex;
  double theWeight;
};
#endif

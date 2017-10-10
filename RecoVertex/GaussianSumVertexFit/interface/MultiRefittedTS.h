#ifndef MultiRefittedTS_H
#define MultiRefittedTS_H

#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

/**
 * Caching refitted state of the trajectory after the vertex fit is
 * done. This is for a multi-state track, where each state is each refitted 
 * state is a single RefittedTrackState, of whatever parametrization
 */

class Surface;

class MultiRefittedTS : public RefittedTrackState<5> {

public:

  typedef ReferenceCountingPointer<RefittedTrackState<5> > RefCountedRefittedTrackState;
  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

  /**
   *   Constructor with a reference surface, to be used to assemble the 
   *   TSOS components on one identical surface.
   */
  MultiRefittedTS(const std::vector<RefCountedRefittedTrackState> & prtsComp,
	const Surface & referenceSurface);

  /**
   *   Constructor with a reference position. The surface which is going to be usedto assemble the 
   *   TSOS components will be the surface perpendicular to the PCA of the state with the highest weight
   *   to the reference point.
   */
  MultiRefittedTS(const std::vector<RefCountedRefittedTrackState> & prtsComp,
	const GlobalPoint & referencePosition);

  ~MultiRefittedTS() override{}

  /**
   * Returns a FreeTrajectoryState. It will be the FTS of the single, collapsed
   * state.
   */

  FreeTrajectoryState freeTrajectoryState() const override;

  /**
   * Returns a multi-state TSOS at a given surface
   */
  TrajectoryStateOnSurface trajectoryStateOnSurface(
  		const Surface & surface) const override;

  /**
   * Returns a multi-state TSOS at a given surface, with a given propagator
   */

  TrajectoryStateOnSurface trajectoryStateOnSurface(
		const Surface & surface, const Propagator & propagator) const override;

  /**
   *   Returns a new reco::Track, which can then be made persistent. The parameters are taken
   *    from FTS described above.
   */

  reco::TransientTrack transientTrack() const override;

  /**
   * Vector containing the refitted track parameters. Not possible yet for a
   * multi-state, throws an exception.
   */

  AlgebraicVectorN  parameters() const override;

  /**
   * The covariance matrix. Not possible yet for a
   * multi-state, throws an exception.
   */

  AlgebraicSymMatrixNN  covariance() const override;

  /**
   * Position at which the momentum is defined. Not possible yet for a
   * multi-state, throws an exception.
   */

  GlobalPoint position() const override;

  /**
   * Vector containing the parameters describing the momentum as the vertex.
   * These are (signed transverse curvature, theta, phi). Not possible yet for a
   * multi-state, throws an exception.
   */

  AlgebraicVectorM momentumVector() const override;

  double weight() const override;

  std::vector<ReferenceCountingPointer<RefittedTrackState<5> > > components() const override
  {
    return theComponents;
  }

  /**
   * This method is meant to returns a new refitted state of the same type, 
   * but with another weight. As we can have several components, each component
   * of the new multi-state will be reweighted so that the sum of all weights
   * is equal to the specified weight.
   * The current state is unchanged.
   */

  ReferenceCountingPointer<RefittedTrackState<5> > stateWithNewWeight
  	(const double newWeight) const override;


private:

  void computeFreeTrajectoryState() const;


  typedef std::vector<RefCountedRefittedTrackState > RTSvector;

  mutable RTSvector theComponents;
  mutable bool totalWeightAvailable, ftsAvailable;
  mutable double totalWeight;
  mutable FreeTrajectoryState fts;
  const GlobalPoint refPosition;
  ConstReferenceCountingPointer<Surface> refSurface;
  const bool surf;

};
#endif

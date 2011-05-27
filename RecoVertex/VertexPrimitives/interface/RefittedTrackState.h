#ifndef RefittedTrackState_H
#define RefittedTrackState_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <vector>

/**
 * Pure abstract base class, caching refitted state of the
 * trajectory after the vertex fit is done. Every concrete implementaton
 * deals with its own trajectory representation.
 */
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class Surface;
class Propagator;

template <unsigned int N>
class RefittedTrackState : public ReferenceCounted {

public:

//   typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicMatrixN3;
  typedef ROOT::Math::SVector<double,N> AlgebraicVectorN;
  typedef ROOT::Math::SVector<double,N-2> AlgebraicVectorM;
//   typedef ROOT::Math::SMatrix<double,N,3,ROOT::Math::MatRepStd<double,N,3> > AlgebraicMatrixN3;
//   typedef ROOT::Math::SMatrix<double,N-2,3,ROOT::Math::MatRepStd<double,N-2,3> > AlgebraicMatrixM3;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;

  virtual ~RefittedTrackState(){}

  /**
   * Transformation into a FreeTrajectoryState
   */
  virtual FreeTrajectoryState freeTrajectoryState() const = 0;

  /**
   * Transformation into a TSOS at a given surface
   */
  virtual TrajectoryStateOnSurface trajectoryStateOnSurface(
  		const Surface & surface) const = 0;

  /**
   * Transformation into a TSOS at a given surface, with a given propagator
   */

  virtual TrajectoryStateOnSurface trajectoryStateOnSurface(
		const Surface & surface, const Propagator & propagator) const = 0;

  /**
   * Vector containing the refitted track parameters.
   */
  virtual AlgebraicVectorN parameters() const = 0;

  /**
   * The covariance matrix
   */
  virtual AlgebraicSymMatrixNN covariance() const = 0;

  /**
   * Position at which the momentum is defined.
   */
  virtual GlobalPoint position() const = 0;

  /**
   * Vector containing the parameters describing the momentum as the vertex
   */

  virtual AlgebraicVectorM momentumVector() const = 0;

  /**
   *   The weight of this component in a mixture
   */

  virtual double weight() const = 0;

  /**
   * Returns a new refitted state of the same type, but with another weight.
   * The current state is unchanged.
   */

  virtual ReferenceCountingPointer<RefittedTrackState> stateWithNewWeight
  	(const double newWeight) const = 0;

  virtual std::vector< ReferenceCountingPointer<RefittedTrackState> > components() const = 0;

  virtual reco::TransientTrack transientTrack() const = 0;

};
#endif

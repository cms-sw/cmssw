#ifndef PerigeeMultiLTS_H
#define PerigeeMultiLTS_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include <vector>

/**
 * Multi-state version of PerigeeLinearizedTrackState
 */

class PerigeeMultiLTS : public LinearizedTrackState<5> {


public:

  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

  /** Friend class properly dealing with creation
   *  of reference-counted pointers to LinearizedTrack objects
   */
  friend class MultiPerigeeLTSFactory;

  /**
   * Returns a new linearized state with respect to a new linearization point.
   * A new object of the same type is returned, without change to the existing one.
   */

  virtual  RefCountedLinearizedTrackState stateWithNewLinearizationPoint
  	(const GlobalPoint & newLP) const;


  /**
   * The point at which the track state has been linearized
   */
  const GlobalPoint & linearizationPoint() const { return theLinPoint; }

  virtual reco::TransientTrack track() const { return theTrack; }

  /**
   * Returns theoriginal (multi-state) TrajectoryStateOnSurface whith which
   * this instance has been created with.
   */
  const TrajectoryStateOnSurface state() const { return theTSOS; }

  /** Method returning the constant term of the Taylor expansion
   *  of the measurement equation for the collapsed track state
   */
  virtual const AlgebraicVectorN & constantTerm() const;

  /** Method returning the Position Jacobian from the Taylor expansion
   *  (Matrix A) for the collapsed track state
   */
  virtual const AlgebraicMatrixN3 & positionJacobian() const;

  /** Method returning the Momentum Jacobian from the Taylor expansion
   *  (Matrix B) for the collapsed track state
   */
  virtual const AlgebraicMatrixNM & momentumJacobian() const;

  /** Method returning the parameters of the Taylor expansion for
   *  the collapsed track state
   */
  virtual const AlgebraicVectorN & parametersFromExpansion() const;

  /** Method returning the track state at the point of closest approach
   *  to the linearization point, in the transverse plane (a.k.a.
   *  transverse impact point),  for the collapsed track state
   */
  const TrajectoryStateClosestToPoint & predictedState() const;

  /** Method returning the parameters of the track state at the
   *  transverse impact point, for the collapsed track state
   */
  AlgebraicVectorN predictedStateParameters() const;

  /** Method returning the momentum part of the parameters of the track state
   *  at the linearization point, for the collapsed track state
   */
  virtual AlgebraicVectorM predictedStateMomentumParameters() const;

  /** Method returning the weight matrix of the track state at the
   *  transverse impact point, for the collapsed track state
   * The error variable is 0 in case of success.
   */
  AlgebraicSymMatrixNN predictedStateWeight(int & error) const;

  /** Method returning the covariance matrix of the track state at the
   *  transverse impact point, for the collapsed track state
   */
  AlgebraicSymMatrixNN predictedStateError() const;

  /** Method returning the momentum covariance matrix of the track state at the
   *  transverse impact point, for the collapsed track state
   */
  AlgebraicSymMatrixMM predictedStateMomentumError() const;

  TrackCharge charge() const {return theCharge;}

  bool hasError() const;

  bool operator ==(LinearizedTrackState<5>& other)const;

  /** Creates the correct refitted state according to the results of the
   *  track refit.
   */
  virtual RefCountedRefittedTrackState createRefittedTrackState(
  	const GlobalPoint & vertexPosition,
	const AlgebraicVectorM & vectorParameters,
	const AlgebraicSymMatrixOO & covarianceMatrix) const;

  virtual AlgebraicVector5 refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const;

  virtual void inline checkParameters(AlgebraicVector5 & parameters) const;
  /**
   * The weight of this state. It will be the sum of the weights of the
   * individual components in the mixture.
   */

  virtual double weightInMixture() const {return theTSOS.weight();}

  /**
   * Vector of individual components in the mixture.
   */

  virtual std::vector<ReferenceCountingPointer<LinearizedTrackState<5> > > components()
  				const {return ltComp;}

private:

  /** Constructor with the linearization point and the track.
   *  Private, can only be used by LinearizedTrackFactory.
   * \param linP	The linearization point
   * \param track	The RecTrack
   * \param tsos	The original (multi-state) TrajectoryStateOnSurface
   */
   PerigeeMultiLTS(const GlobalPoint & linP, const reco::TransientTrack & track,
  	const TrajectoryStateOnSurface& tsos);

  /** Linearize the collapsed track state.
   */
  void prepareCollapsedState() const;

  GlobalPoint theLinPoint;
  reco::TransientTrack theTrack;


  const TrajectoryStateOnSurface theTSOS;
  std::vector<RefCountedLinearizedTrackState> ltComp;
  mutable RefCountedLinearizedTrackState collapsedStateLT;
  LinearizedTrackStateFactory theLTSfactory;
  TrackCharge theCharge;
  mutable bool collapsedStateAvailable;
};

#endif

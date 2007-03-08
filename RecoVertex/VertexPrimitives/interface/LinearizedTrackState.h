#ifndef LinearizedTrackState_H
#define LinearizedTrackState_H

//#include "CommonReco/CommonVertex/interface/ImpactPointMeasurement.h"

//#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/VertexPrimitives/interface/RefCountedRefittedTrackState.h"
#include <vector>

/**
 *
 *  Abstract base class for all kind of linearizedtrack like objects.
 *  Calculates and stores the ImpactPointMeasurement of the
 *  impact point (point of closest approach in 3D) to the
 *  given linearization point.
 *  (see V.Karimaki, HIP-1997-77 / EXP)
 *
 *  Computes the parameters of the trajectory at the transverse
 *  point of closest approach (in the global transverse plane) to
 *  the linearization point, and the jacobiam matrices.
 *  (see R.Fruehwirth et al. Data Analysis Techniques in HEP Experiments
 *  Second Edition, Cambridge University Press 2000, or
 *  R.Fruehwirth et al. Vertex reconstruction and track bundling at the LEP
 *  collider using robust algorithms. Computer Physics Communications 96
 *  (1996) 189-208).
 */

class LinearizedTrackState : public ReferenceCounted {

public:

  virtual ~LinearizedTrackState(){}

  /**
   * Returns a new linearized state with respect to a new linearization point.
   * A new object of the same type is returned, without change to the existing one.
   */
  virtual  ReferenceCountingPointer<LinearizedTrackState> stateWithNewLinearizationPoint
    (const GlobalPoint & newLP) const = 0;

  /** Access methods
   */
  virtual const GlobalPoint & linearizationPoint() const = 0;

  /** Method returning the constant term of the Taylor expansion
   *  of the measurement equation
   */
  virtual AlgebraicVector constantTerm() const = 0;

  /** Method returning the Position Jacobian from the Taylor expansion
   *  (Matrix A)
   */
  virtual AlgebraicMatrix positionJacobian() const = 0;

  /** Method returning the Momentum Jacobian from the Taylor expansion
   *  (Matrix B)
   */
  virtual AlgebraicMatrix momentumJacobian() const = 0;

  /** Method returning the parameters of the Taylor expansion
   */
  virtual AlgebraicVector parametersFromExpansion() const = 0;

  /** Method returning the parameters of the track state at the
   *  linearization point.
   */
  virtual AlgebraicVector predictedStateParameters() const = 0;

  /** Method returning the momentum part of the parameters of the track state
   *  at the linearization point.
   */
  virtual AlgebraicVector predictedStateMomentumParameters() const = 0;

  /** Method returning the weight matrix of the track state at the
   *  linearization point.
   */
  virtual AlgebraicSymMatrix predictedStateWeight() const = 0;

  /** Method returning the momentum covariance matrix of the track state at the
   *  transverse impact point.
   */
  virtual AlgebraicSymMatrix predictedStateMomentumError() const = 0;

  /** Method returning the covariance matrix of the track state at the
   *  linearization point.
   */
  virtual AlgebraicSymMatrix predictedStateError() const = 0;

  virtual bool hasError() const = 0;

  virtual TrackCharge charge() const = 0;

  /** Method returning the impact point measurement
   */
//   virtual ImpactPointMeasurement impactPointMeasurement() const = 0;

  virtual bool operator ==(LinearizedTrackState& other) const = 0;

  /** Creates the correct refitted state according to the results of the
   *  track refit.
   */
  virtual RefCountedRefittedTrackState createRefittedTrackState(
  	const GlobalPoint & vertexPosition,
	const AlgebraicVector & vectorParameters,
	const AlgebraicSymMatrix & covarianceMatrix) const = 0;

  /** Method returning the parameters of the Taylor expansion evaluated with the 
   *  refitted state.
   */
  virtual AlgebraicVector refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const;

  virtual double weightInMixture() const = 0;

  virtual std::vector< ReferenceCountingPointer<LinearizedTrackState> > components() 
  								const = 0;

  virtual reco::TransientTrack track() const = 0;



};

#endif

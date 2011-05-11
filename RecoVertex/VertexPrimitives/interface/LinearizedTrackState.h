#ifndef LinearizedTrackState_H
#define LinearizedTrackState_H

//#include "CommonReco/CommonVertex/interface/ImpactPointMeasurement.h"

//#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "Math/SMatrix.h"
#include "DataFormats/CLHEP/interface/Migration.h"

#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
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

template <unsigned int N>
class LinearizedTrackState : public ReferenceCounted {

public:

  typedef ROOT::Math::SVector<double,N> AlgebraicVectorN;
  typedef ROOT::Math::SVector<double,N-2> AlgebraicVectorM;
  typedef ROOT::Math::SMatrix<double,N,3,ROOT::Math::MatRepStd<double,N,3> > AlgebraicMatrixN3;
  typedef ROOT::Math::SMatrix<double,N,N-2,ROOT::Math::MatRepStd<double,N,N-2> > AlgebraicMatrixNM;
  typedef ROOT::Math::SMatrix<double,N-2,3,ROOT::Math::MatRepStd<double,N-2,3> > AlgebraicMatrixM3;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepSym<double,N-2> > AlgebraicSymMatrixMM;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepSym<double,N+1> > AlgebraicSymMatrixOO;

  typedef ReferenceCountingPointer<RefittedTrackState<N> > RefCountedRefittedTrackState;

  virtual ~LinearizedTrackState(){}

  /**
   * Returns a new linearized state with respect to a new linearization point.
   * A new object of the same type is returned, without change to the existing one.
   */
  virtual  ReferenceCountingPointer<LinearizedTrackState<N> > stateWithNewLinearizationPoint
    (const GlobalPoint & newLP) const = 0;

  /** Access methods
   */
  virtual const GlobalPoint & linearizationPoint() const = 0;

  /** Method returning the constant term of the Taylor expansion
   *  of the measurement equation
   */
  virtual const AlgebraicVectorN & constantTerm() const = 0;

  /** Method returning the Position Jacobian from the Taylor expansion
   *  (Matrix A)
   */
  virtual const AlgebraicMatrixN3 & positionJacobian() const = 0;

  /** Method returning the Momentum Jacobian from the Taylor expansion
   *  (Matrix B)
   */
  virtual const AlgebraicMatrixNM & momentumJacobian() const = 0;

  /** Method returning the parameters of the Taylor expansion
   */
  virtual const AlgebraicVectorN & parametersFromExpansion() const = 0;

  /** Method returning the parameters of the track state at the
   *  linearization point.
   */
  virtual AlgebraicVectorN predictedStateParameters() const = 0;

  /** Method returning the momentum part of the parameters of the track state
   *  at the linearization point.
   */
  virtual AlgebraicVectorM predictedStateMomentumParameters() const = 0;

  /** Method returning the weight matrix of the track state at the
   *  linearization point.
   * The error variable is 0 in case of success.
   */
  virtual AlgebraicSymMatrixNN predictedStateWeight(int & error) const = 0;

  /** Method returning the momentum covariance matrix of the track state at the
   *  transverse impact point.
   */
  virtual AlgebraicSymMatrixMM predictedStateMomentumError() const = 0;

  /** Method returning the covariance matrix of the track state at the
   *  linearization point.
   */
  virtual AlgebraicSymMatrixNN predictedStateError() const = 0;

  virtual bool hasError() const = 0;

  virtual TrackCharge charge() const = 0;

  /** Method returning the impact point measurement
   */
//   virtual ImpactPointMeasurement impactPointMeasurement() const = 0;

  virtual bool operator ==(LinearizedTrackState<N> & other) const = 0;

  /** Creates the correct refitted state according to the results of the
   *  track refit.
   */
  virtual RefCountedRefittedTrackState createRefittedTrackState(
  	const GlobalPoint & vertexPosition,
	const AlgebraicVectorM & vectorParameters,
	const AlgebraicSymMatrixOO & covarianceMatrix) const = 0;

  /** Method returning the parameters of the Taylor expansion evaluated with the
   *  refitted state.
   */
  virtual AlgebraicVectorN refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const = 0;

  virtual inline void checkParameters(AlgebraicVectorN & parameters) const
	{}

  virtual double weightInMixture() const = 0;

  virtual std::vector< ReferenceCountingPointer<LinearizedTrackState<N> > > components()
  								const = 0;

  virtual reco::TransientTrack track() const = 0;

  virtual bool isValid() const { return true; }
};

#endif

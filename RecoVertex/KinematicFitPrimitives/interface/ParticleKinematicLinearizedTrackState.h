#ifndef ParticleKinematicLinearizedTrackState_H
#define ParticleKinematicLinearizedTrackState_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/PerigeeKinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicStateBuilder.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/CLHEP/interface/Migration.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

class ParticleKinematicLinearizedTrackState : public LinearizedTrackState<6> {

public:
 
 friend class ParticleKinematicLinearizedTrackStateFactory;
 typedef ReferenceCountingPointer<LinearizedTrackState<6> > RefCountedLinearizedTrackState;

 ParticleKinematicLinearizedTrackState()
 {jacobiansAvailable = false;}
 
 /**
  * Returns a new linearized state with respect to a new linearization point.
  * A new object of the same type is returned, without change to the existing one.
  */ 
 ReferenceCountingPointer<LinearizedTrackState<6> > stateWithNewLinearizationPoint
  	(const GlobalPoint & newLP) const override; 
	
 /**
  * The point at which the track state has been linearized
  */	
 const GlobalPoint & linearizationPoint() const override { return theLinPoint; }
 

/** Method returning the constant term of the Taylor expansion
 *  of measurement equation
 */
 const AlgebraicVector6 & constantTerm() const override;

/** Method returning the Position Jacobian from the Taylor expansion 
 *  (Matrix A)
 */
 const AlgebraicMatrix63 & positionJacobian() const override;

/** Method returning the Momentum Jacobian from the Taylor expansion 
 *  (Matrix B)
 */   
 const AlgebraicMatrix64 & momentumJacobian() const override;

/** Method returning the parameters of the Taylor expansion
 */
 const AlgebraicVector6 & parametersFromExpansion() const override;

/** Method returning the track state at the point of closest approach 
 *  to the linearization point, in the transverse plane (a.k.a. 
 *  transverse impact point). 
 */      

/**
 * extended perigee predicted parameters
 */
 AlgebraicVector6 predictedStateParameters() const override;
    
/**
 * returns predicted 4-momentum in extended perigee parametrization
 */  
 AlgebraicVectorM predictedStateMomentumParameters() const override; 

/**
 * 4x4 error matrix ofe xtended perigee mometum components
 */ 
 AlgebraicSymMatrix44 predictedStateMomentumError() const override; 

/**
 * Full predicted weight matrix
 */  
 AlgebraicSymMatrix66 predictedStateWeight(int & error) const override;
  
/**
 * Full predicted error matrix
 */  
 AlgebraicSymMatrix66 predictedStateError() const override; 

// /** 
//  * Method returning the impact point measurement     
//  */      
//  ImpactPointMeasurement impactPointMeasurement() const;
  
 TrackCharge charge() const override;
 
 RefCountedKinematicParticle particle() const;
   
 bool operator ==(LinearizedTrackState<6>& other)const override;
 
 bool hasError() const override;
 
 RefCountedRefittedTrackState createRefittedTrackState(
                     const GlobalPoint & vertexPosition, 
		     const AlgebraicVectorM & vectorParameters,
		     const AlgebraicSymMatrix77 & covarianceMatrix) const override;
		     
  /** Method returning the parameters of the Taylor expansion evaluated with the
   *  refitted state.
   */
  AlgebraicVectorN refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const override;

  inline void checkParameters(AlgebraicVectorN & parameters) const override;
		     
 double weightInMixture() const override;

 std::vector<ReferenceCountingPointer<LinearizedTrackState<6> > > components()
  									const override;
 
 reco::TransientTrack track() const override;

 
private:

/** Constructor with the linearization point and the track. 
 *  Private, can only be used by LinearizedTrackFactory.
 */     
  ParticleKinematicLinearizedTrackState(const GlobalPoint & linP, RefCountedKinematicParticle & prt) 
              : theLinPoint(linP), part(prt), jacobiansAvailable(false),
	      theCharge(prt->currentState().particleCharge())  ,impactPointAvailable(false)
	   
    {}
    
/** 
 * Method calculating the track parameters and the Jacobians.
 */
  void computeJacobians() const;
/**
 * Method calculating the track parameters and the Jacobians for charged particles.
 */
  void computeChargedJacobians() const;
 
/**
 * Method calculating the track parameters and the Jacobians for neutral particles.
 */ 
  void computeNeutralJacobians() const;   
     
     
  GlobalPoint theLinPoint;
  RefCountedKinematicParticle part;
  TransientTrackKinematicStateBuilder builder;
  
  mutable bool errorAvailable;
  mutable bool jacobiansAvailable;
  mutable AlgebraicMatrix63 thePositionJacobian;
  mutable AlgebraicMatrix64 theMomentumJacobian;
  mutable PerigeeKinematicState thePredState;
  mutable AlgebraicVector6 theConstantTerm;
  mutable AlgebraicVector6 theExpandedParams;
  
  TrackCharge theCharge;
//   ImpactPointMeasurementExtractor theIPMExtractor;
  mutable bool impactPointAvailable; 
};
#endif

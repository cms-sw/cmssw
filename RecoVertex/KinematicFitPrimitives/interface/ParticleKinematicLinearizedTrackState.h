#ifndef ParticleKinematicLinearizedTrackState_H
#define ParticleKinematicLinearizedTrackState_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
// #include "CommonReco/CommonVertex/interface/ImpactPointMeasurement.h"
// #include "CommonReco/CommonVertex/interface/ImpactPointMeasurementExtractor.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedRefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/PerigeeKinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicStateBuilder.h"

class ParticleKinematicLinearizedTrackState : public LinearizedTrackState{

public:
 
 friend class ParticleKinematicLinearizedTrackStateFactory;
 
 ParticleKinematicLinearizedTrackState()
 {jacobiansAvailable = false;}
 
 /**
  * Returns a new linearized state with respect to a new linearization point.
  * A new object of the same type is returned, without change to the existing one.
  */ 
 virtual ReferenceCountingPointer<LinearizedTrackState> stateWithNewLinearizationPoint
  	(const GlobalPoint & newLP) const; 
	
 /**
  * The point at which the track state has been linearized
  */	
 const GlobalPoint & linearizationPoint() const { return theLinPoint; }
 

/** Method returning the constant term of the Taylor expansion
 *  of measurement equation
 */
 AlgebraicVector constantTerm() const;

/** Method returning the Position Jacobian from the Taylor expansion 
 *  (Matrix A)
 */
 AlgebraicMatrix positionJacobian() const;

/** Method returning the Momentum Jacobian from the Taylor expansion 
 *  (Matrix B)
 */   
 AlgebraicMatrix momentumJacobian() const;

/** Method returning the parameters of the Taylor expansion
 */
 AlgebraicVector parametersFromExpansion() const;

/** Method returning the track state at the point of closest approach 
 *  to the linearization point, in the transverse plane (a.k.a. 
 *  transverse impact point). 
 */      

/**
 * extended perigee predicted parameters
 */
 AlgebraicVector predictedStateParameters() const;
    
/**
 * returns predicted 4-momentum in extended perigee parametrization
 */  
 AlgebraicVector predictedStateMomentumParameters() const; 

/**
 * 4x4 error matrix ofe xtended perigee mometum components
 */ 
 AlgebraicSymMatrix predictedStateMomentumError() const; 

/**
 * Full predicted weight matrix
 */  
 AlgebraicSymMatrix predictedStateWeight() const;
  
/**
 * Full predicted error matrix
 */  
 AlgebraicSymMatrix predictedStateError() const; 

// /** 
//  * Method returning the impact point measurement     
//  */      
//  ImpactPointMeasurement impactPointMeasurement() const;
  
 TrackCharge charge() const;
 
 RefCountedKinematicParticle particle() const;
   
 bool operator ==(LinearizedTrackState& other)const;
 
 bool hasError() const;
 
 RefCountedRefittedTrackState createRefittedTrackState(
                     const GlobalPoint & vertexPosition, 
	             const AlgebraicVector & vectorParameters,
	             const AlgebraicSymMatrix & covarianceMatrix)const; 
		     
		     
 double weightInMixture() const;

 std::vector<ReferenceCountingPointer<LinearizedTrackState> > components()
  									const;
 
 virtual reco::TransientTrack track() const;

 
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
  mutable AlgebraicMatrix thePositionJacobian, theMomentumJacobian;
  mutable PerigeeKinematicState thePredState;
  mutable AlgebraicVector theConstantTerm;
  mutable AlgebraicVector theExpandedParams;
  
  TrackCharge theCharge;
//   ImpactPointMeasurementExtractor theIPMExtractor;
  mutable bool impactPointAvailable; 
};
#endif

#ifndef KinematicParticle_H
#define KinematicParticle_H

#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

/**
 * Abstract base class for KinematicParticles
 * created out of Physics objects of different types
 * Common access methods are implemented here.
 * All the virtual abstract methods should be 
 * implemented by user.
 */
 
class KinematicConstraint;
class KinematicTree;

class KinematicParticle : public ReferenceCounted 
{
public:

  typedef ReferenceCountingPointer<LinearizedTrackState<6> > RefCountedLinearizedTrackState;

 friend class KinematicParticleVertexFitter;
 friend class KinematicTree;

/**
 * Default constructor: does not create
 * a valid particle. Method is needed for
 * debugging purposes only.
 */  
 KinematicParticle(){}
 
 virtual ~KinematicParticle();
							   
/**
 * Comparison by contents operators
 * Returns TRUE if initial PhysicsObjects
 * match(if they exist). If not, 
 * compares the initial KinematicStates
 * Retunes true if they match.
 * Should be implemented by user
 */ 
 virtual bool operator==(const KinematicParticle& other)const = 0;

 virtual bool operator==(const ReferenceCountingPointer<KinematicParticle>& other) const = 0;

 virtual bool operator!=(const KinematicParticle& other) const = 0;
  
/**
 * Comparison by adress operator
 * Has NO physical meaning
 * To be used inside graph only
 */ 
 virtual bool operator<(const KinematicParticle& other)const;
  
/**
 * Access to the kinematic state
 * with which particle was first created
 */ 
 virtual KinematicState initialState()const;
 
/**
 * Access to the last calculated kinematic state
 */ 
 virtual KinematicState currentState()const;
 
/**
 * Access to KinematicState of particle
 * at given point. The current state of particle 
 * does not change after this operation.
 */ 
 virtual KinematicState stateAtPoint(const GlobalPoint& point)const = 0;
 
/**
 * Method producing new particle out of the current 
 * one and RefittedState obtained from kinematic fitting.
 * To be used by Fitter classes only. Method should be
 * implemented by used for every specific type of KinematicParticle
 */ 
 virtual ReferenceCountingPointer<KinematicParticle> refittedParticle(const KinematicState& state,
                               float chi2, float ndf, KinematicConstraint * cons = 0) const = 0;
			       
/**
 * Method returning LinearizedTrackState of the particle needed for
 * Kalman flter vertex fit. Should be implemented by user.  For track(helix)-like
 * objects one can use the ParticleLinearizedTrackStateFactory class.
 */			       
 virtual RefCountedLinearizedTrackState particleLinearizedTrackState(const GlobalPoint& point)const = 0;			       
  
/**
 * Returns last constraint aplied to
 * this particle.
 */  
 virtual KinematicConstraint * lastConstraint() const;
 
/**
 * Returns the state of Kinematic Particle before
 * last constraint was aplied
 */    
 virtual ReferenceCountingPointer<KinematicParticle> previousParticle() const;
  
/**
 * Returns the pointer to the kinematic
 * tree (if any) current particle belongs to
 * 0 pointer
 * returned in case  not any tree is built yet
 */  
 virtual KinematicTree * correspondingTree() const;
  
/**
 * Access metods for chi2 and
 * number of degrees of freedom
 */  
 virtual float chiSquared() const;
  
 virtual float degreesOfFreedom() const;
   
  const MagneticField* magneticField() const {return theField;}

  reco::TransientTrack refittedTransientTrack() const;

protected: 

 virtual void setTreePointer(KinematicTree * tr) const;
  
/**
 * Data members which should be initialized by user in 
 * derived classes
 */ 

  const MagneticField* theField;

//pointer to the tree current
//particle  belongs to
 mutable  KinematicTree * tree;
 
//last constraint applied 
 mutable KinematicConstraint * lConstraint;

//previous particle
 mutable ReferenceCountingPointer<KinematicParticle> pState;
 
//initial kinematic state of
//current particle
 KinematicState initState; 
 
//particle state at point  
 mutable KinematicState cState;
  
//chi2 and number of degrees of freedom
 float chi2;
  
 float ndf;  
};
#endif

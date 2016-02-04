#ifndef VirtualKinematicParticleFactory_H
#define VirtualKinematicParticleFactory_H

#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicStateBuilder.h"
#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"

class VirtualKinematicParticleFactory
{
public:

/**
 * Factory constructor taking a KinematicStatePropagator
 */ 
  VirtualKinematicParticleFactory();   
 
/**
 * Factory constructor taking a KinematicStatePropagator
 */ 
  VirtualKinematicParticleFactory(KinematicStatePropagator * pr);                                                        
  
/**
 * Default destructor
 */ 
  ~VirtualKinematicParticleFactory()
  {delete propagator;}

/**
 * Method building a particle out of new created kinematic state,
 * chi2, number of degrees of freedom and history information
 */
  RefCountedKinematicParticle particle(const KinematicState& kineState, float& chiSquared,
                 float& degreesOfFr, ReferenceCountingPointer<KinematicParticle> previousParticle,
				           KinematicConstraint * lastConstraint = 0)const;
private:
  
 KinematicStatePropagator * propagator;
 const TransientTrackKinematicStateBuilder builder;

};
#endif

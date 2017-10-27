#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
 
 
VirtualKinematicParticleFactory::VirtualKinematicParticleFactory()
{propagator =  new TrackKinematicStatePropagator();}
 
VirtualKinematicParticleFactory::VirtualKinematicParticleFactory(KinematicStatePropagator * pr)
{
 if(pr!=nullptr)
 {
  propagator = pr->clone();
 }else{
  propagator =  new TrackKinematicStatePropagator();
 }
}
 
RefCountedKinematicParticle VirtualKinematicParticleFactory::particle(const KinematicState& kineState, 
          float& chiSquared, float& degreesOfFr, ReferenceCountingPointer<KinematicParticle> previousParticle,
			                	       KinematicConstraint * lastConstraint)const
{
 if(previousParticle.get() != nullptr)
 {
  KinematicParticle * prp = &(*previousParticle);
  VirtualKinematicParticle * pr = dynamic_cast<VirtualKinematicParticle * >(prp);
  if(pr == nullptr){ throw VertexException("KinematicParticleFactoryFromTransientTrack::Previous particle passed is not TransientTrack based!");}
 } 
 return ReferenceCountingPointer<KinematicParticle>(new VirtualKinematicParticle(kineState, chiSquared, degreesOfFr, 
                                                                            lastConstraint, previousParticle, propagator));
}

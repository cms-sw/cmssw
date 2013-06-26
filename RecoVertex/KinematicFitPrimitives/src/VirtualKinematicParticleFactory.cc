#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
 
 
VirtualKinematicParticleFactory::VirtualKinematicParticleFactory()
{propagator =  new TrackKinematicStatePropagator();}
 
VirtualKinematicParticleFactory::VirtualKinematicParticleFactory(KinematicStatePropagator * pr)
{
 if(pr!=0)
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
 if(previousParticle.get() != 0)
 {
  KinematicParticle * prp = &(*previousParticle);
  VirtualKinematicParticle * pr = dynamic_cast<VirtualKinematicParticle * >(prp);
  if(pr == 0){ throw VertexException("KinematicParticleFactoryFromTransientTrack::Previous particle passed is not TransientTrack based!");}
 } 
 return ReferenceCountingPointer<KinematicParticle>(new VirtualKinematicParticle(kineState, chiSquared, degreesOfFr, 
                                                                            lastConstraint, previousParticle, propagator));
}

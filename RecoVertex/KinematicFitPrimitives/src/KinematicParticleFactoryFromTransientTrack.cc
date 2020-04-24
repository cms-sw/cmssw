#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"

KinematicParticleFactoryFromTransientTrack::KinematicParticleFactoryFromTransientTrack()
{propagator = new TrackKinematicStatePropagator();}


KinematicParticleFactoryFromTransientTrack::KinematicParticleFactoryFromTransientTrack(KinematicStatePropagator * pr)
{
 if(pr != nullptr)
 {
  propagator = pr->clone();
 }else{
  propagator = new TrackKinematicStatePropagator();
 }
}

RefCountedKinematicParticle KinematicParticleFactoryFromTransientTrack::particle(const reco::TransientTrack& initialTrack, 
                                                                           const ParticleMass& massGuess,
									   float chiSquared, 
									   float degreesOfFr, 
									   float& m_sigma) const
{
// cout<<"calling the state builder"<<endl;
 KinematicState initState = builder(initialTrack,massGuess, m_sigma);
 const reco::TransientTrack * track = &initialTrack;
 KinematicConstraint * lastConstraint = nullptr;
 ReferenceCountingPointer<KinematicParticle> previousParticle = nullptr;
 return ReferenceCountingPointer<KinematicParticle>(new TransientTrackKinematicParticle(initState,
                      chiSquared,degreesOfFr, lastConstraint, previousParticle, propagator, track));
}

RefCountedKinematicParticle KinematicParticleFactoryFromTransientTrack::particle(const reco::TransientTrack& initialTrack, 
                                                                           const ParticleMass& massGuess,
                                                                           float chiSquared, 
                                                                           float degreesOfFr, 
                                                                           float& m_sigma,
                                                                           const FreeTrajectoryState &freestate) const
{
// cout<<"calling the state builder"<<endl;
 KinematicState initState = builder(freestate,massGuess, m_sigma);
 const reco::TransientTrack * track = &initialTrack;
 KinematicConstraint * lastConstraint = nullptr;
 ReferenceCountingPointer<KinematicParticle> previousParticle = nullptr;
 return ReferenceCountingPointer<KinematicParticle>(new TransientTrackKinematicParticle(initState,
                      chiSquared,degreesOfFr, lastConstraint, previousParticle, propagator, track));
}

RefCountedKinematicParticle KinematicParticleFactoryFromTransientTrack::particle(const reco::TransientTrack& initialTrack, 
                                                                           const ParticleMass& massGuess,
									   float chiSquared, 
									   float degreesOfFr,
		                                                           const GlobalPoint& expPoint, 
			                                                   float m_sigma) const
{
//  FreeTrajectoryState st(initialTrack.impactPointTSCP().theState());
 KinematicState initState = builder(initialTrack.impactPointTSCP().theState(), massGuess, m_sigma, expPoint); 
 const reco::TransientTrack * track = &initialTrack;
 KinematicConstraint * lastConstraint = nullptr;
 ReferenceCountingPointer<KinematicParticle> previousParticle = nullptr;
 return ReferenceCountingPointer<KinematicParticle>(new TransientTrackKinematicParticle(initState,
                                chiSquared,degreesOfFr, lastConstraint, previousParticle, propagator, track));
}
 

RefCountedKinematicParticle KinematicParticleFactoryFromTransientTrack::particle(const KinematicState& kineState, 
                                                                           float& chiSquared,
                                                                           float& degreesOfFr,
	                                          ReferenceCountingPointer<KinematicParticle> previousParticle,
							           KinematicConstraint * lastConstraint) const
{
 const reco::TransientTrack * track;
 KinematicParticle * prp = &(*previousParticle);
// FIXME
//  if(previousParticle.isValid()){
   TransientTrackKinematicParticle * pr = dynamic_cast<TransientTrackKinematicParticle * >(prp);
   if(pr == nullptr){
    throw VertexException("KinematicParticleFactoryFromTransientTrack::Previous particle passed is not TransientTrack based!");
   }else{track  = pr->initialTransientTrack();}
//  }else{track = 0;}
 return ReferenceCountingPointer<KinematicParticle>(new TransientTrackKinematicParticle(kineState,
                                chiSquared, degreesOfFr, lastConstraint, previousParticle, propagator, track));
}

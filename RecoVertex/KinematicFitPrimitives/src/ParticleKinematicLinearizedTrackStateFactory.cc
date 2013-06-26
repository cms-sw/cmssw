#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackStateFactory.h"
                               
ParticleKinematicLinearizedTrackStateFactory::RefCountedLinearizedTrackState 
ParticleKinematicLinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, 
                                            RefCountedKinematicParticle & prt) const
{
  return RefCountedLinearizedTrackState(new ParticleKinematicLinearizedTrackState(linP, prt));
}  

const ParticleKinematicLinearizedTrackStateFactory * ParticleKinematicLinearizedTrackStateFactory::clone() const
{
  return new ParticleKinematicLinearizedTrackStateFactory ( *this );
}

ParticleKinematicLinearizedTrackStateFactory::RefCountedLinearizedTrackState 
ParticleKinematicLinearizedTrackStateFactory::linearizedTrackState(
	const GlobalPoint & linP, const reco::TransientTrack & track) const
{
throw VertexException("ParticleKinematicLinearizedTrackStateFactory from TransientTrack not possible");
}
ParticleKinematicLinearizedTrackStateFactory::RefCountedLinearizedTrackState 
ParticleKinematicLinearizedTrackStateFactory::linearizedTrackState(
	const GlobalPoint & linP, const reco::TransientTrack & track,
	const TrajectoryStateOnSurface& tsos) const
{
throw VertexException("ParticleKinematicLinearizedTrackStateFactory from TransientTrack not possible");
}


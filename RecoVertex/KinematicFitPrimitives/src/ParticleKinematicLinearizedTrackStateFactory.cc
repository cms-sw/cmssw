#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackStateFactory.h"
                               
RefCountedLinearizedTrackState 
ParticleKinematicLinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, 
                                            RefCountedKinematicParticle & prt) const
{
  return RefCountedLinearizedTrackState(new ParticleKinematicLinearizedTrackState(linP, prt));
}  

#ifndef ParticleKinematicLinearizedTrackStateFactory_H
#define ParticleKinematicLinearizedTrackStateFactory_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackState.h"

class ParticleKinematicLinearizedTrackStateFactory{

/**
 * Class building LinearizedTrackState needed for
 * Kalman based vertex fit out of track(helix like) KinematicParticle
 */

public:

/**
 * Deafault constructor and desructor
 */
 ParticleKinematicLinearizedTrackStateFactory(){}
 
 ~ParticleKinematicLinearizedTrackStateFactory(){}

/**
 * Method constructing KinearizedTrackState out of
 * KinematicParticle and Linearization point.
 */
 RefCountedLinearizedTrackState linearizedTrackState(const GlobalPoint & linP,
                                     RefCountedKinematicParticle & prt) const;  
};
#endif

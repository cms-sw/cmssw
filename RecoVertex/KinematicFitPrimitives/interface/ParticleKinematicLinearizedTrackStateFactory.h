#ifndef ParticleKinematicLinearizedTrackStateFactory_H
#define ParticleKinematicLinearizedTrackStateFactory_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/AbstractLTSFactory.h"

class ParticleKinematicLinearizedTrackStateFactory : public AbstractLTSFactory<6>{

/**
 * Class building LinearizedTrackState needed for
 * Kalman based vertex fit out of track(helix like) KinematicParticle
 */

public:

/**
 * Method constructing KinearizedTrackState out of
 * KinematicParticle and Linearization point.
 */
 RefCountedLinearizedTrackState linearizedTrackState(const GlobalPoint & linP,
                                     RefCountedKinematicParticle & prt) const;  

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track) const override;

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track,
    	const TrajectoryStateOnSurface& tsos) const override;

  const ParticleKinematicLinearizedTrackStateFactory * clone() const override;

};
#endif

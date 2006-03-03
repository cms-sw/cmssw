#ifndef LinearizedTrackStateFactory_H
#define LinearizedTrackStateFactory_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"


/**
 *  Concrete class to encapsulate the creation of a RefCountedLinearizedTrack,
 *  which is a reference-counting pointer.
 *  Should always be used in order to create a new RefCountedLinearizedTrack,
 *  so that the reference-counting mechanism works well.
 */

class LinearizedTrackStateFactory {

public:

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track) const;

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track,
    	const TrajectoryStateOnSurface& tsos) const;

  RefCountedLinearizedTrackState
    linearizedTrackState(LinearizedTrackState * lts) const;

//   RefCountedLinearizedTrackState
//     linearizedTrackState(const GlobalPoint & linP, RefCountedKinematicParticle & prt) const;

};

#endif

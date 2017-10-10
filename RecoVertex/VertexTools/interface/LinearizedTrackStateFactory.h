#ifndef LinearizedTrackStateFactory_H
#define LinearizedTrackStateFactory_H

#include "RecoVertex/VertexTools/interface/AbstractLTSFactory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

/**
 *  Concrete class to encapsulate the creation of a RefCountedLinearizedTrack,
 *  which is a reference-counting pointer.
 *  Should always be used in order to create a new RefCountedLinearizedTrack,
 *  so that the reference-counting mechanism works well.
 */

class LinearizedTrackStateFactory : public AbstractLTSFactory<5> {

public:

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track) const override;

  RefCountedLinearizedTrackState
    linearizedTrackState(const GlobalPoint & linP, const reco::TransientTrack & track,
    	const TrajectoryStateOnSurface& tsos) const override;

  RefCountedLinearizedTrackState
    linearizedTrackState(LinearizedTrackState<5> * lts) const;

  const LinearizedTrackStateFactory * clone() const override;

//   RefCountedLinearizedTrackState
//     linearizedTrackState(const GlobalPoint & linP, RefCountedKinematicParticle & prt) const;

};

#endif

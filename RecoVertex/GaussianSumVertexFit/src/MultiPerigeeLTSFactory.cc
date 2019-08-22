#include "RecoVertex/GaussianSumVertexFit/interface/MultiPerigeeLTSFactory.h"

// MultiPerigeeLTSFactory::MultiPerigeeLTSFactory()
// {
// }
//
// MultiPerigeeLTSFactory::~MultiPerigeeLTSFactory()
// {
// }

MultiPerigeeLTSFactory::RefCountedLinearizedTrackState MultiPerigeeLTSFactory::linearizedTrackState(
    const GlobalPoint& linP, const reco::TransientTrack& track, const TrajectoryStateOnSurface& tsos) const {
  return RefCountedLinearizedTrackState(new PerigeeMultiLTS(linP, track, tsos));
}

MultiPerigeeLTSFactory::RefCountedLinearizedTrackState MultiPerigeeLTSFactory::linearizedTrackState(
    const GlobalPoint& linP, const reco::TransientTrack& track) const {
  return RefCountedLinearizedTrackState(new PerigeeMultiLTS(linP, track, track.stateOnSurface(linP)));
}

const MultiPerigeeLTSFactory* MultiPerigeeLTSFactory::clone() const { return new MultiPerigeeLTSFactory(*this); }

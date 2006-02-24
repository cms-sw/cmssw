#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"


RefCountedLinearizedTrackState 
LinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, 
	const DummyRecTrack & track, const TrajectoryStateOnSurface& tsos) const
{
  return RefCountedLinearizedTrackState(
    new PerigeeLinearizedTrackState(linP, track, tsos ) );
}
 
RefCountedLinearizedTrackState 
LinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, 
  					const DummyRecTrack & track) const
{
  return RefCountedLinearizedTrackState(
    new PerigeeLinearizedTrackState(linP, track, track.impactPointState() ) );
}
 
RefCountedLinearizedTrackState
LinearizedTrackStateFactory::linearizedTrackState
				(LinearizedTrackState * lts) const
{
  return RefCountedLinearizedTrackState(lts);
}    
// RefCountedLinearizedTrackState
// LinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, RefCountedKinematicParticle & prt) const
// {
//  return RefCountedLinearizedTrackState(new KinematicLinearizedTrackState(linP, prt));
// }

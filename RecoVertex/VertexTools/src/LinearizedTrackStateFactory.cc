#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"


LinearizedTrackStateFactory::RefCountedLinearizedTrackState 
LinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, 
	const reco::TransientTrack & track, const TrajectoryStateOnSurface& tsos) const
{
  return RefCountedLinearizedTrackState(
    new PerigeeLinearizedTrackState(linP, track, tsos ) );
}

LinearizedTrackStateFactory::RefCountedLinearizedTrackState 
LinearizedTrackStateFactory::linearizedTrackState(const GlobalPoint & linP, 
  					const reco::TransientTrack & track) const
{
  return RefCountedLinearizedTrackState(
    new PerigeeLinearizedTrackState(linP, track, track.impactPointState() ) );
}
 
LinearizedTrackStateFactory::RefCountedLinearizedTrackState
LinearizedTrackStateFactory::linearizedTrackState
				(LinearizedTrackState<5> * lts) const
{
  return RefCountedLinearizedTrackState(lts);
}    

const LinearizedTrackStateFactory * LinearizedTrackStateFactory::clone() const
{
  return new LinearizedTrackStateFactory ( *this );
}


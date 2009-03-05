#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include <algorithm>
using namespace std;
using namespace reco;

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const GlobalPoint& priorPos,
	const GlobalError & priorError)
{ 
  VertexState priorVertexState(priorPos, priorError);
  return constrain(track, priorVertexState);
}


SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const TransientTrack & track,  const VertexState priorVertexState)
{
  // Linearize tracks

  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef VertexTrack<5>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  RefCountedLinearizedTrackState lTrData 
      = theLTrackFactory.linearizedTrackState(priorVertexState.position(), track);
  RefCountedVertexTrack vertexTrack =  theVTrackFactory.vertexTrack(lTrData, priorVertexState);

  // Fit vertex

  vector<RefCountedVertexTrack> initialTracks;
  CachingVertex<5> vertex(priorVertexState,priorVertexState,initialTracks,0);
  vertex = vertexUpdator.add(vertex, vertexTrack);
  if (!vertex.isValid()) {
    return BTFtuple(false, TransientTrack(), 0.);
  }
  RefCountedVertexTrack nTrack = theVertexTrackUpdator.update(vertex, vertexTrack);
  return BTFtuple(true, nTrack->refittedState()->transientTrack(), nTrack->smoothedChi2());
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const GlobalPoint& priorPos,
	const GlobalError& priorError)
{ 
  return constrain(ttFactory.build(fts), priorPos, priorError);
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const reco::BeamSpot & spot )
{
  VertexState priorVertexState(spot);
  return constrain(track, priorVertexState);
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const reco::BeamSpot & spot)
{ 
  VertexState priorVertexState(spot);
  return constrain(ttFactory.build(fts), priorVertexState);
}


#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

#include <algorithm>
using namespace std;
using namespace reco;

TransientTrack SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const GlobalPoint& priorPos,
	const GlobalError & priorError) const
{ 
  VertexState priorVertexState(priorPos, priorError);

  // Linearize tracks

  RefCountedLinearizedTrackState lTrData 
      = theLTrackFactory.linearizedTrackState(priorPos, track);
  RefCountedVertexTrack vertexTrack =  theVTrackFactory.vertexTrack(lTrData,priorVertexState);

  // Fit vertex

  vector<RefCountedVertexTrack> initialTracks;
  CachingVertex vertex(priorVertexState,priorVertexState,initialTracks,0);
  vertex = vertexUpdator.add(vertex, vertexTrack);
  RefCountedVertexTrack nTrack = theVertexTrackUpdator.update(vertex, vertexTrack);

  return nTrack->refittedState()->transientTrack() ;
}

TransientTrack SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const GlobalPoint& priorPos,
	const GlobalError& priorError) const
{ 
  return constrain(ttFactory.build(fts), priorPos, priorError);
}

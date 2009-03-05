#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include <algorithm>
using namespace std;
using namespace reco;

bool SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const GlobalPoint& priorPos,
	const GlobalError & priorError)
{ 
  VertexState priorVertexState(priorPos, priorError);
  return constrain(track, priorVertexState);
}


bool SingleTrackVertexConstraint::constrain(
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
    validity_ = false;
    return false;
  }
  RefCountedVertexTrack nTrack = theVertexTrackUpdator.update(vertex, vertexTrack);
  validity_ = true;
  result_ = TrackFloatPair(nTrack->refittedState()->transientTrack(), nTrack->smoothedChi2()) ;
  return validity_;
}

bool SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const GlobalPoint& priorPos,
	const GlobalError& priorError)
{ 
  return constrain(ttFactory.build(fts), priorPos, priorError);
}

bool SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const reco::BeamSpot & spot )
{
  VertexState priorVertexState(spot);
  return constrain(track, priorVertexState);
}

bool SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const reco::BeamSpot & spot)
{ 
  VertexState priorVertexState(spot);
  return constrain(ttFactory.build(fts), priorVertexState);
}

SingleTrackVertexConstraint::TrackFloatPair SingleTrackVertexConstraint::result() const
{
  if (!validity_) throw VertexException("SingleTrackVertexConstraint::constaint had files. Check validity first!");
  return result_;
}

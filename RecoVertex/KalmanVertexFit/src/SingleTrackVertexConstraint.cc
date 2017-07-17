#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h" 

#include <algorithm>
using namespace std;
using namespace reco;

namespace {
  // FIXME
  // hard-coded tracker bounds
  // workaround while waiting for Geometry service
  static const float TrackerBoundsRadius = 112;
  static const float TrackerBoundsHalfLength = 273.5;
  bool insideTrackerBounds(const GlobalPoint& point) {
    return ((point.transverse() < TrackerBoundsRadius)
        && (abs(point.z()) < TrackerBoundsHalfLength));
  }
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const GlobalPoint& priorPos,
	const GlobalError & priorError) const
{ 
  VertexState priorVertexState(priorPos, priorError);
  return constrain(track, priorVertexState);
}


SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const TransientTrack & track,  const VertexState priorVertexState) const
{
  // Linearize tracks

  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef VertexTrack<5>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  double field  = track.field()->inInverseGeV(track.impactPointState().globalPosition()).z();
  int  nominalBfield  = track.field()->nominalValue();
  if ((fabs(field) < 1e-4)&&(fabs(nominalBfield)!=0)) { //protection for the case where the magnet is off
      LogDebug("RecoVertex/SingleTrackVertexConstraint") 
	 << "Initial state is very far, field is close to zero (<1e-4): " << field << "\n";
      return BTFtuple(false, TransientTrack(), 0.);
  }

  RefCountedLinearizedTrackState lTrData 
      = theLTrackFactory.linearizedTrackState(priorVertexState.position(), track);
  RefCountedVertexTrack vertexTrack =  theVTrackFactory.vertexTrack(lTrData, priorVertexState);

  // Fit vertex

  std::vector<RefCountedVertexTrack> initialTracks;
  CachingVertex<5> vertex(priorVertexState,priorVertexState,initialTracks,0);
  vertex = vertexUpdator.add(vertex, vertexTrack);
  if (!vertex.isValid()) {
    return BTFtuple(false, TransientTrack(), 0.);
  } else  if (doTrackerBoundCheck_ && (!insideTrackerBounds(vertex.position()))) {
      LogDebug("RecoVertex/SingleTrackVertexConstraint") 
	 << "Fitted position is out of tracker bounds.\n";
      return BTFtuple(false, TransientTrack(), 0.);
  }

  RefCountedVertexTrack nTrack = theVertexTrackUpdator.update(vertex, vertexTrack);
  return BTFtuple(true, nTrack->refittedState()->transientTrack(), nTrack->smoothedChi2());
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const GlobalPoint& priorPos,
	const GlobalError& priorError) const
{ 
  return constrain(ttFactory.build(fts), priorPos, priorError);
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const TransientTrack & track, const reco::BeamSpot & spot ) const
{
  VertexState priorVertexState(spot);
  return constrain(track, priorVertexState);
}

SingleTrackVertexConstraint::BTFtuple SingleTrackVertexConstraint::constrain(
	const FreeTrajectoryState & fts, const reco::BeamSpot & spot) const
{ 
  VertexState priorVertexState(spot);
  return constrain(ttFactory.build(fts), priorVertexState);
}


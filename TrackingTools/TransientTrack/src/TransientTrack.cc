#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

using namespace reco;

TransientTrack::TransientTrack( const Track & tk ) : Track(tk), tk_(tk) {
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), covariance(), GlobalPoint(0.,0.,0.));
}

TransientTrack::TransientTrack( const TransientTrack & tt ) :
  Track(tt.persistentTrack()), tk_(tt.persistentTrack())
{originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), covariance(), GlobalPoint(0.,0.,0.));}

TransientTrack& TransientTrack::operator=(const TransientTrack & tt)
{
  if (this == &tt) return *this;
  Track::operator=(tk_);
//   tk_ = tt.persistentTrack();
// 
//   originalTSCP = TrajectoryStateClosestToPoint
//     (parameters(), covariance(), GlobalPoint(0.,0.,0.));
  return *this;
}


TrajectoryStateOnSurface TransientTrack::impactPointState() const
{
  if (!stateAtVertexAvailable) calculateStateAtVertex();
  return theStateAtVertex;
}

void TransientTrack::calculateStateAtVertex() const
{
  theStateAtVertex = TransverseImpactPointExtrapolator().extrapolate(
     originalTSCP.theState(), originalTSCP.position());
  stateAtVertexAvailable = true;
}


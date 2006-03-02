#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

using namespace reco;

TransientTrack::TransientTrack( const Track & tk ) : Track(tk), tk_(tk) {
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), covariance(), GlobalPoint(0.,0.,0.));
}

TrajectoryStateOnSurface TransientTrack::impactPointState()
{
  if (!stateAtVertexAvailable) calculateStateAtVertex();
  return theStateAtVertex;
}

void TransientTrack::calculateStateAtVertex()
{
  theStateAtVertex = TransverseImpactPointExtrapolator().extrapolate(
     originalTSCP.theState(), originalTSCP.position());
  stateAtVertexAvailable = true;
}


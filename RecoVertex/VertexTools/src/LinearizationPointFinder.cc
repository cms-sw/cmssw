#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
GlobalPoint LinearizationPointFinder::getLinearizationPoint(
    const std::vector<FreeTrajectoryState> & ftses ) const
{
  std::vector < reco::TransientTrack > rectracks;
  TransientTrackFromFTSFactory factory;
  for ( std::vector< FreeTrajectoryState>::const_iterator fts=ftses.begin();
        fts!=ftses.end() ; ++fts )
  rectracks.push_back ( factory.build(*fts));
  return getLinearizationPoint(rectracks);
}

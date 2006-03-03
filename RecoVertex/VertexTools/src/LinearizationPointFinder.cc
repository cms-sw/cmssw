#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
// #include "CommonReco/PatternTools/interface/ConcreteRecTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

GlobalPoint LinearizationPointFinder::getLinearizationPoint(
    const std::vector<FreeTrajectoryState> & ftses ) const
{
  std::vector < reco::TransientTrack > rectracks;
//   vector < ConcreteRecTrack > concreteTks;
// 
//   TrajectorySeed ts;
//   vector < TrajectoryMeasurement > tm;
//   for ( vector< FreeTrajectoryState>::const_iterator fts=ftses.begin();
//         fts!=ftses.end() ; ++fts )
//   {   
//    GlobalPoint fPos = fts->position();
//    ConcreteRecTrack ctrack ( *fts, fPos, 0., 0, 0, ts, tm );
//    concreteTks.push_back(ctrack);
//   }
// 
//   for (vector< ConcreteRecTrack >::const_iterator itk = concreteTks.begin(); 
//        itk != concreteTks.end(); itk++) {
//     RecTrack ntrack(&(*itk));
//     rectracks.push_back ( ntrack );
//   }

  return getLinearizationPoint(rectracks);
}

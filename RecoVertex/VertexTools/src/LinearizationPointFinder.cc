#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
GlobalPoint LinearizationPointFinder::getLinearizationPoint(const std::vector<FreeTrajectoryState>& ftses) const {
  std::vector<reco::TransientTrack> rectracks;
  TransientTrackFromFTSFactory factory;
  for (const auto& ftse : ftses)
    rectracks.push_back(factory.build(ftse));
  return getLinearizationPoint(rectracks);
}

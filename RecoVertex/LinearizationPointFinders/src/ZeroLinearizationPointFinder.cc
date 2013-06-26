#include "RecoVertex/LinearizationPointFinders/interface/ZeroLinearizationPointFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

GlobalPoint ZeroLinearizationPointFinder::getLinearizationPoint(
    const std::vector<FreeTrajectoryState> & tracks ) const
{
  return GlobalPoint(0.,0.,0.);
}

GlobalPoint ZeroLinearizationPointFinder::getLinearizationPoint(
    const std::vector<reco::TransientTrack> & tracks ) const
{
  return GlobalPoint(0.,0.,0.);
}

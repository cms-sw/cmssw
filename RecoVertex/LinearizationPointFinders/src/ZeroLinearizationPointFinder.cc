#include "RecoVertex/LinearizationPointFinders/interface/ZeroLinearizationPointFinder.h"
#include "RecoVertex/VertexPrimitives/interface/DummyRecTrack.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

GlobalPoint ZeroLinearizationPointFinder::getLinearizationPoint(
    const vector<FreeTrajectoryState> & tracks ) const
{
  return GlobalPoint(0.,0.,0.);
};

GlobalPoint ZeroLinearizationPointFinder::getLinearizationPoint(
    const vector<DummyRecTrack> & tracks ) const
{
  return GlobalPoint(0.,0.,0.);
}

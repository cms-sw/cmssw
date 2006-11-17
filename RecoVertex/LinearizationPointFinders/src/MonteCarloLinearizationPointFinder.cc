#if 0

#include "RecoVertex/LinearizationPointFinders/interface/MonteCarloLinearizationPointFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

MonteCarloLinearizationPointFinder::MonteCarloLinearizationPointFinder() :
    thePt ( GlobalPoint(0.,0.,0.) ) {}

void MonteCarloLinearizationPointFinder::setPoint ( const GlobalPoint & pos )
{
  thePt = pos;
}

GlobalPoint MonteCarloLinearizationPointFinder::getLinearizationPoint(
    const std::vector<FreeTrajectoryState> & tracks ) const
{
  return getLinearizationPoint ( std::vector < reco::TransientTrack > () );
}

GlobalPoint MonteCarloLinearizationPointFinder::getLinearizationPoint(
    const std::vector<reco::TransientTrack> & tracks ) const
{
  // cout << "[MonteCarloLinearizationPointFinder] point " << thePt << endl;
  return thePt;
}

#endif

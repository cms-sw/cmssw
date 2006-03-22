#include "RecoVertex/LinearizationPointFinders/interface/MonteCarloLinearizationPointFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

MonteCarloLinearizationPointFinder::MonteCarloLinearizationPointFinder() :
    thePt ( GlobalPoint(0.,0.,0.) ) {};

void MonteCarloLinearizationPointFinder::setPoint ( const GlobalPoint & pos )
{
  thePt = pos;
};

GlobalPoint MonteCarloLinearizationPointFinder::getLinearizationPoint(
    const vector<FreeTrajectoryState> & tracks ) const
{
  return getLinearizationPoint ( vector < reco::TransientTrack > () );
};

GlobalPoint MonteCarloLinearizationPointFinder::getLinearizationPoint(
    const vector<reco::TransientTrack> & tracks ) const
{
  // cout << "[MonteCarloLinearizationPointFinder] point " << thePt << endl;
  return thePt;
}

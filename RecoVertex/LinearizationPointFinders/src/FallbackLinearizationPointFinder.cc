#include "RecoVertex/LinearizationPointFinders/interface/FallbackLinearizationPointFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoVertex/VertexTools/interface/FsmwModeFinder3d.h"

FallbackLinearizationPointFinder::FallbackLinearizationPointFinder ( 
    const ModeFinder3d & m ) : theModeFinder ( m.clone() ) {}

GlobalPoint FallbackLinearizationPointFinder::getLinearizationPoint(
    const std::vector<FreeTrajectoryState> & tracks ) const
{
  return GlobalPoint(0.,0.,0.);
}

GlobalPoint FallbackLinearizationPointFinder::getLinearizationPoint(
    const std::vector<reco::TransientTrack> & tracks ) const
{
    switch ( tracks.size() )
    {
      case 0:
        return GlobalPoint ( 0.,0.,0. );
      case 1:
        return tracks.begin()->impactPointState().globalPosition();
      default:
      {
        std::vector < std::pair < GlobalPoint, float > > wtracks;
        wtracks.reserve ( tracks.size() - 1 );
        for ( std::vector< reco::TransientTrack >::const_iterator i=tracks.begin(); 
              i!=tracks.end() ; ++i )
        {
            std::pair < GlobalPoint, float > tmp ( 
                i->impactPointState().globalPosition(), 1. );
            wtracks.push_back ( tmp );
        }
        return (*theModeFinder) ( wtracks );
      }
    }
  return GlobalPoint ( 0.,0.,0. );
}

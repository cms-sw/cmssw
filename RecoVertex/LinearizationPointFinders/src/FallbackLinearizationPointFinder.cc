#include "RecoVertex/LinearizationPointFinders/interface/FallbackLinearizationPointFinder.h"
#include "RecoVertex/VertexPrimitives/interface/DummyRecTrack.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "RecoVertex/VertexTools/interface/FsmwModeFinder3d.h"

FallbackLinearizationPointFinder::FallbackLinearizationPointFinder ( 
    const ModeFinder3d & m ) : theModeFinder ( m.clone() ) {};

GlobalPoint FallbackLinearizationPointFinder::getLinearizationPoint(
    const vector<FreeTrajectoryState> & tracks ) const
{
  return GlobalPoint(0.,0.,0.);
};

GlobalPoint FallbackLinearizationPointFinder::getLinearizationPoint(
    const vector<DummyRecTrack> & tracks ) const
{
  try {
    switch ( tracks.size() )
    {
      case 0:
        return GlobalPoint ( 0.,0.,0. );
      case 1:
        return tracks.begin()->innermostState().globalPosition();
      default:
      {
        vector < pair < GlobalPoint, float > > wtracks;
        wtracks.reserve ( tracks.size() - 1 );
        for ( vector< DummyRecTrack >::const_iterator i=tracks.begin(); 
              i!=tracks.end() ; ++i )
        {
          try {
            pair < GlobalPoint, float > tmp ( 
                i->innermostState().globalPosition(), 1. );
            wtracks.push_back ( tmp );
          } catch (...) {};
        };
        return (*theModeFinder) ( wtracks );
      };
    };
  } catch ( ... ) {};
  return GlobalPoint ( 0.,0.,0. );
}

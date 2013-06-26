#include "RecoVertex/MultiVertexFit/interface/LinTrackCache.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"

using namespace std;

namespace
{
  int verbose()
  {
    static const int ret = 0; /* SimpleConfigurable<int>
      (0, "LinTrackCache:Debug").value(); */
    return ret;
  }

  float maxRelinDistance()
  {
    // maximum distance before relinearizing
    static const float ret = 1e-2; /*  SimpleConfigurable<float>
      (0.01, "LinTrackCache:RelinearizeAfter").value();
      */
    return ret;
  }
}

LinTrackCache::RefCountedLinearizedTrackState LinTrackCache::linTrack
    ( const GlobalPoint & pos, const reco::TransientTrack & rt )
{
  if ( theHasLinTrack[pos][rt] )
  {
    return theLinTracks[pos][rt];
  };

  LinearizedTrackStateFactory lTrackFactory;
  RefCountedLinearizedTrackState lTrData =
    lTrackFactory.linearizedTrackState( pos, rt );

  theLinTracks[pos][rt]=lTrData;
  theHasLinTrack[pos][rt]=true;
  return lTrData;
}

bool LinTrackCache::Comparer::operator() ( const GlobalPoint & left,                                               const GlobalPoint & right )
{
  // if theyre closer than 1 micron, they're
  // indistinguishable, i.e. the same
  // static const double max = 1e-4 * 1e-4;
  // if ( ( left - right ).mag2() < max ) return false;

  if ( left.x() != right.x() )
  {
    return ( left.x() < right.x() );
  } else if (left.y() != right.y()) {
    return ( left.y() < right.y() );
  } else {
    return ( left.z() < right.z() );
  }
}

bool LinTrackCache::Vicinity::operator() ( const GlobalPoint & p1,
                                           const GlobalPoint & p2 )
{
  if ( (p1 - p2).mag() < maxRelinDistance() )
  {
    return false;
  };
  return Comparer()(p1,p2);
}

LinTrackCache::~LinTrackCache()
{
  clear();
}

void LinTrackCache::clear()
{
  theLinTracks.clear();
  theHasLinTrack.clear();
}

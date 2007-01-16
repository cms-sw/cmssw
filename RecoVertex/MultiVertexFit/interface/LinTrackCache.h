#ifndef _LinTrackCache_H_
#define _LinTrackCache_H_

#include <map>
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class LinTrackCache
{
private:
  struct Comparer
  {
    bool operator() ( const GlobalPoint &, const GlobalPoint & );
  };

  struct Vicinity
  {
    bool operator() ( const GlobalPoint &, const GlobalPoint & );
  };

public:
  /**
   *  \class LinTrackCache
   *  caches LinearizedTrackStates
   */
  RefCountedLinearizedTrackState linTrack ( const GlobalPoint &, const reco::TransientTrack & );
  ~LinTrackCache();
  void clear();

private:
  typedef std::map < reco::TransientTrack, RefCountedLinearizedTrackState > LinTrkMap;
  typedef std::map < reco::TransientTrack, bool > HasLinTrkMap;
  std::map < GlobalPoint, LinTrkMap, Vicinity > theLinTracks;
  std::map < GlobalPoint, HasLinTrkMap, Vicinity > theHasLinTrack;
};

#endif

#ifndef _LinTrackCache_H_
#define _LinTrackCache_H_

#include <map>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class LinTrackCache
{
private:
  struct Comparer
  {
    bool operator() ( const GlobalPoint &, const GlobalPoint & ) const;
  };

  struct Vicinity
  {
    bool operator() ( const GlobalPoint &, const GlobalPoint & ) const;
  };

public:

  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

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

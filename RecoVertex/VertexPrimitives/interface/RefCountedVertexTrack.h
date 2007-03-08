#ifndef RefCountedVertexTrack_H
#define RefCountedVertexTrack_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"

typedef ReferenceCountingPointer<VertexTrack> RefCountedVertexTrack;

class VertexTrackEqual {
  public:
    VertexTrackEqual( const RefCountedVertexTrack & t) : track_( t ) { }
    bool operator()( const RefCountedVertexTrack & t ) const { return t->operator==(*track_);}
  private:
    const RefCountedVertexTrack & track_;
};


#endif

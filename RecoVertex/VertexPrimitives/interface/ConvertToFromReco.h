#ifndef RecoVertex_ConvertToFromReco_h
#define RecoVertex_ConvertToFromReco_h

#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"

namespace RecoVertex{
  static reco::Vertex::Point convertPos(const GlobalPoint& p) 
    {
      reco::Vertex::Point pos;
      pos.SetX(p.x());
      pos.SetY(p.y());
      pos.SetZ(p.z());
      return pos;
    }
  static GlobalPoint convertPos(const reco::Vertex::Point& p)
    { return GlobalPoint(p.x(), p.y(), p.z()); }
}

#endif

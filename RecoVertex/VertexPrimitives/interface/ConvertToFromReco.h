#ifndef RecoVertex_ConvertToFromReco_h
#define RecoVertex_ConvertToFromReco_h
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace RecoVertex{
  static reco::Vertex::Error convertError(const GlobalError& ge) 
    {
      reco::Vertex::Error error;
      error(0,0) = ge.cxx();
      error(0,1) = ge.cyx();
      error(0,2) = ge.czx();
      error(1,1) = ge.cyy();
      error(1,2) = ge.czy();
      error(2,2) = ge.czz();
      return error;
    }
  static GlobalError convertError(const reco::Vertex::Error& error)
    { return GlobalError(error(0,0), error(0,1), error(1,1), error(0,2), error(1,2), error(2,2)); }
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

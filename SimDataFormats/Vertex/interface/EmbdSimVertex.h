#ifndef EmbdSimVertex_H
#define EmbdSimVertex_H

#include "SimDataFormats/Vertex/interface/CoreSimVertex.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>
class EmbdSimVertex : public CoreSimVertex
{
 public:
  
  typedef edm::Ref<std::vector<EmbdSimVertex> > EmbdSimVertexRef ;
  typedef edm::RefProd<std::vector<EmbdSimVertex> > EmbdSimVertexRefProd;
  typedef edm::RefVector<std::vector<EmbdSimVertex> > EmbdSimVertexRefVector;
  typedef CoreSimVertex Core;
  /// constructor
    EmbdSimVertex();
    EmbdSimVertex(const Hep3Vector & v, float tof);
    /// full constructor (position, time, index of parent in final vector)
    EmbdSimVertex(const Hep3Vector & v, float tof, int it);
    /// constructor from transient
    EmbdSimVertex(const CoreSimVertex & t, int it);
    /// index of the parent in the Event SimTrack container (-1 if no parent)
    int parentIndex() const { return  itrack; }
    bool noParent() const { return  itrack==-1; }
private: 
    int itrack;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const EmbdSimVertex& v);
 

#endif

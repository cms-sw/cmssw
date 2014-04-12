#ifndef SimVertex_H
#define SimVertex_H

#include "SimDataFormats/Vertex/interface/CoreSimVertex.h"
class SimVertex : public CoreSimVertex
{

 public:
  
  typedef CoreSimVertex Core;
  /// constructor
  SimVertex();

  /// constructor from transient
  SimVertex(const CoreSimVertex & t, int it);

  SimVertex( const math::XYZVectorD& v, float tof ) ;

  /// full constructor (position, time, index of parent in final vector)
  SimVertex( const math::XYZVectorD& v, float tof, int it ) ;

  /// constructor from transient
  SimVertex(const CoreSimVertex & t, int it, unsigned int vId );

  SimVertex( const math::XYZVectorD& v, float tof, unsigned int vId ) ;

  /// full constructor (position, time, index of parent in final vector)
  SimVertex( const math::XYZVectorD& v, float tof, int it, unsigned int vId ) ;


  /// G4 TrackId of the parent in the Event SimTrack container (-1 if no parent)
  /// BE CAREFUL this is not a vector index
  int parentIndex() const { return  itrack; }
  bool noParent() const { return  itrack==-1; }

  void setVertexId(unsigned int n) {vtxId = n;}
  unsigned int vertexId() const { return  vtxId; }  

  void setProcessType(unsigned int ty) {procType = ty;}
  unsigned int processType() const { return procType; }  

private: 
  int itrack;
  unsigned int vtxId; 
  unsigned int procType;
};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const SimVertex& v);
 

#endif

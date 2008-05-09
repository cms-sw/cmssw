#ifndef SimVertex_H
#define SimVertex_H

#include "SimDataFormats/Vertex/interface/CoreSimVertex.h"
class SimVertex : public CoreSimVertex
{

 public:
  
  typedef CoreSimVertex Core;
  /// constructor
  SimVertex();
  SimVertex( const math::XYZVectorD& v, float tof ) ;

  /// full constructor (position, time, index of parent in final vector)
  SimVertex( const math::XYZVectorD& v, float tof, int it ) ;

  /// constructor from transient
  SimVertex(const CoreSimVertex & t, int it);

  /// G4 TrackId of the parent in the Event SimTrack container (-1 if no parent)
  /// BE CAREFUL this is not a vector index
  int parentIndex() const { return  itrack; }
  bool noParent() const { return  itrack==-1; }

private: 
  int itrack;

};

#include <iosfwd>
std::ostream & operator <<(std::ostream & o , const SimVertex& v);
 

#endif

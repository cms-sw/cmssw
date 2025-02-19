#ifndef KinematicVertexAssociator_H
#define KinematicVertexAssociator_H

#include <vector>
#include "TrackerReco/TkEvent/interface/TkSimVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
/**
 * A prototype class for associators linking the
 * KinematicVertices to the SimVertices. Must be generalized later,
 * when KinematicVertex becomes an instance of the Vertex class.
 *
 * Kirill Prokofiev, October 2004
 */


class TkSimVertex;
//class RefCountedKinematicVertex;

class KinematicVertexAssociator {
public:
    
  typedef std::vector<RefCountedKinematicVertex> KinematicVertexContainer;
  typedef std::vector<const TkSimVertex *> SimVertexPtrContainer;   
  typedef std::vector<const TkSimVertex> SimVertexContainer;   

  virtual ~KinematicVertexAssociator() {}

  virtual SimVertexPtrContainer simVertices(const RefCountedKinematicVertex&) const = 0;
  
  virtual KinematicVertexContainer recVertices(const TkSimVertex&) const = 0;

  virtual KinematicVertexAssociator * clone() const = 0;
  
};


#endif

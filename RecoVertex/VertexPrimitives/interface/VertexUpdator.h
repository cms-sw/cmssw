#ifndef VertexUpdator_H
#define VertexUpdator_H

#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

/** 
 * Pure abstract base class for VertexUpdators. 
 * Updates CachingVertex with one VertexTrack. 
 */

class VertexUpdator {

public:
  
  /**
   * Default Constructor
   */  
   VertexUpdator() {}
    
   virtual ~VertexUpdator() {}
    
  /**
   * Method updating the vertex, with the information contained 
   * in the track.
   */
  virtual CachingVertex add(const CachingVertex & v,
			    const RefCountedVertexTrack t) const = 0;

  virtual CachingVertex remove(const CachingVertex & v,
			       const RefCountedVertexTrack t) const = 0;

  virtual VertexUpdator * clone() const = 0;  

};


#endif

#ifndef KinematicVertexFactory_H
#define KinematicVertexFactory_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"

/**
 * Factory to create Reference counting pointers
 * to KinematicVertex objects. Can be used both
 * to create object and pointers or simple
 * pointers to existing object. 
 *
 * Kirill Prokofiev December 2002
 */



class KinematicVertexFactory
{
public:
 
 KinematicVertexFactory()
 {}
 
/**
 * Constructor with vertex state, chi2 and ndf.
 * Previous state of the vertex pointer is set to 0.
 */	
 static RefCountedKinematicVertex vertex(const VertexState& state, float totalChiSq, float degreesOfFr) 
 {
  return ReferenceCountingPointer<KinematicVertex>(new KinematicVertex(state,totalChiSq,
                                                                           degreesOfFr));
 } 
 
/**
 * Constructor with previous (before constraint)
 * state of the vertex
 */
  static RefCountedKinematicVertex vertex(const VertexState state, 
                      const ReferenceCountingPointer<KinematicVertex> pVertex,
                                      float totalChiSq, float degreesOfFr)
 {
   return ReferenceCountingPointer<KinematicVertex>(new KinematicVertex(state, pVertex,
                                                             totalChiSq, degreesOfFr));
 }


/**
 * Direct conversion from caching vertex
 */
 static RefCountedKinematicVertex vertex(const CachingVertex<6>& vertex)
 {
  return  ReferenceCountingPointer<KinematicVertex>(new KinematicVertex(vertex)); 
 }

/**
 * Method producing invalid kinematic vertices
 * to mark top production and final state decay vertices
 */ 
 static  RefCountedKinematicVertex vertex()
 {
  return  ReferenceCountingPointer<KinematicVertex>(new KinematicVertex()); 
 }

};
#endif

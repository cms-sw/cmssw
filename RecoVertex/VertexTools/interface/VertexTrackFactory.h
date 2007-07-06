#ifndef VertexTrackFactory_H
#define VertexTrackFactory_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedVertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"

/** 
 *  Concrete class to encapsulate the creation of a RefCountedVertexTrack, 
 *  which is a reference-counting pointer. 
 *  Should always be used in order to create a new RefCountedVertexTrack, 
 *  so that the reference-counting mechanism works well. 
 */ 

class VertexTrackFactory {

public:

  VertexTrackFactory() {}
   ~VertexTrackFactory() {}

  RefCountedVertexTrack
   vertexTrack(const RefCountedLinearizedTrackState lt, 
	       const VertexState vs,
	       float weight = 1.0 ) const {
    return RefCountedVertexTrack(new VertexTrack(lt, vs, weight ));
  };

  RefCountedVertexTrack
   vertexTrack(const RefCountedLinearizedTrackState lt, 
	       const VertexState vs,
	       const RefCountedRefittedTrackState & refittedState, 
	       float smoothedChi2, float weight = 1.0 ) const {
    return RefCountedVertexTrack(new VertexTrack(lt, vs, weight, refittedState,
     				 smoothedChi2));
  };

  RefCountedVertexTrack
  vertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState vs,
	      const RefCountedRefittedTrackState & refittedState,
	      float smoothedChi2,
	      const AlgebraicMatrix & tVCov, float weight = 1.0 ) const {
    return RefCountedVertexTrack(new VertexTrack(lt, vs, weight,
                                 refittedState, smoothedChi2, tVCov));
  };
};

#endif

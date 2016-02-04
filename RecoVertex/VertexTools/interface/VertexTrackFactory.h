#ifndef VertexTrackFactory_H
#define VertexTrackFactory_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"

/** 
 *  Concrete class to encapsulate the creation of a RefCountedVertexTrack, 
 *  which is a reference-counting pointer. 
 *  Should always be used in order to create a new RefCountedVertexTrack, 
 *  so that the reference-counting mechanism works well. 
 */ 

template <unsigned int N>
class VertexTrackFactory {

public:

  typedef ReferenceCountingPointer<RefittedTrackState<N> > RefCountedRefittedTrackState;
  typedef ReferenceCountingPointer<VertexTrack<N> > RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<N> > RefCountedLinearizedTrackState;
  typedef ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > AlgebraicMatrix3M;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepSym<double,N+1> > AlgebraicSymMatrixOO;

  VertexTrackFactory() {}
   ~VertexTrackFactory() {}

  RefCountedVertexTrack
   vertexTrack(const RefCountedLinearizedTrackState lt, 
	       const VertexState vs,
	       float weight = 1.0 ) const {
    return RefCountedVertexTrack(new VertexTrack<N>(lt, vs, weight ));
  };

  RefCountedVertexTrack
   vertexTrack(const RefCountedLinearizedTrackState lt, 
	       const VertexState vs,
	       const RefCountedRefittedTrackState & refittedState, 
	       float smoothedChi2, float weight = 1.0 ) const {
    return RefCountedVertexTrack(new VertexTrack<N>(lt, vs, weight, refittedState,
     				 smoothedChi2));
  };

  RefCountedVertexTrack
  vertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState vs,
	      const RefCountedRefittedTrackState & refittedState,
	      float smoothedChi2,
	      const AlgebraicSymMatrixOO & tVCov, float weight = 1.0 ) const {
    return RefCountedVertexTrack(new VertexTrack<N>(lt, vs, weight,
                                 refittedState, smoothedChi2, tVCov));
  };
};

#endif

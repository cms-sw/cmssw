#ifndef FinalTreeBuilder_H
#define FinalTreeBuilder_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

/**
 * Class building  a resulting output
 * tree out of the information provided
 * by KinematicParticleVertexFitter.
 */
class FinalTreeBuilder{

public:
  FinalTreeBuilder();

  ~FinalTreeBuilder();

  RefCountedKinematicTree buildTree(const CachingVertex<6>& vtx,
                          const std::vector<RefCountedKinematicParticle> &input) const;

private:

 typedef ReferenceCountingPointer<VertexTrack<6> > RefCountedVertexTrack;
 typedef ReferenceCountingPointer<LinearizedTrackState<6> > RefCountedLinearizedTrackState;
 typedef ReferenceCountingPointer<RefittedTrackState<6> > RefCountedRefittedTrackState;

//internal calculation and helper methods
 AlgebraicMatrix momentumPart(const CachingVertex<6>& vtx,
				     const AlgebraicVector7& par)const;

 KinematicVertexFactory * kvFactory;
 VirtualKinematicParticleFactory * pFactory;
};

#endif

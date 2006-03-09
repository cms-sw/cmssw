#ifndef FinalTreeBuilder_H
#define FinalTreeBuilder_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"

/**
 * Class building  a resulting output
 * tree out of the information provided
 * by KinematicParticleVertexFitter.
 */
class FinalTreeBuilder{

public:
  FinalTreeBuilder();
 
  ~FinalTreeBuilder();

  RefCountedKinematicTree buildTree(const CachingVertex& vtx,
                          vector<RefCountedKinematicParticle> input) const;

private:

//internal calculation and helper methods
 AlgebraicMatrix momentumPart(vector<KinematicRefittedTrackState *> rStates,
                                     const CachingVertex& vtx, 
				     const AlgebraicVector& par)const;
				     
 KinematicVertexFactory * kvFactory;				     
 VirtualKinematicParticleFactory * pFactory;				     
};

#endif

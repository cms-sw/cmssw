#ifndef ConstrainedTreeBuilder_H
#define ConstrainedTreeBuilder_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"

/**
 * Class constructing te final output tree  for the constrained vertex fitter. 
 * To be used by corresponding fitter only. Tree builders are scheduled for 
 * generalization: They should be inherited from the single generic class
 * in the next version of the library.
 */

class ConstrainedTreeBuilder
{

public:

 ConstrainedTreeBuilder();
 
 ~ConstrainedTreeBuilder();

/**
 * Method constructing tree out of set of refitted states, vertex, and
 * full covariance matrix.
 */

 RefCountedKinematicTree buildTree(const std::vector<RefCountedKinematicParticle> & initialParticles, 
                         const std::vector<KinematicState> & finalStates,
			 const RefCountedKinematicVertex vtx, const AlgebraicMatrix& fCov) const;

private:

  RefCountedKinematicTree buildTree(const RefCountedKinematicParticle virtualParticle, 
	const RefCountedKinematicVertex vtx, const std::vector<RefCountedKinematicParticle> & particles) const;

  /**
   * Metod to reconstructing the full covariance matrix of the resulting particle.					      
   */
 AlgebraicMatrix covarianceMatrix(const std::vector<RefCountedKinematicParticle> &rPart, 
                                       const AlgebraicVector7& newPar,
				       const AlgebraicMatrix& fitCov)const;
				       
 VirtualKinematicParticleFactory * pFactory;				       
 KinematicVertexFactory * vFactory;
};
#endif

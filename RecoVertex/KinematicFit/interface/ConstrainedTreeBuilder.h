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
 * Method constructing tree out of set of refitted particles, vertex, and
 * full vertex - all tracks covariance matrix. Particles (not states!) are
 * passed since only particle "knows" how to construct itself out of refitted state
 * (which factory/propagator should be used).
 */
 RefCountedKinematicTree buildTree(vector<RefCountedKinematicParticle> prt,
                         RefCountedKinematicVertex vtx,const AlgebraicMatrix& fCov) const;

private:
//Metod reconstructing the full covariance matrix of
//resulting particle. Matrix is returned in two parts:
//first component of vector is p_p covariance;
//the second one is p_x covariance						      
 AlgebraicMatrix momentumPart(vector<RefCountedKinematicParticle> rPart, 
                                       const AlgebraicVector& newPar,
				       const AlgebraicMatrix& fitCov)const;
				       
 VirtualKinematicParticleFactory * pFactory;				       
 KinematicVertexFactory * vFactory;
};
#endif

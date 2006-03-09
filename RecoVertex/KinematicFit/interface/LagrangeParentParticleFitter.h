#ifndef LagrangeParentParticleFitter_H
#define LagrangeParentParticleFitter_H

#include "RecoVertex/KinematicFit/interface/ParentParticleFitter.h"

/**
 * KinematicParticle refit via LMS minimization with Lagrange multipliers
 * method. Tunable for number of iterations and stopping condition.
 * Corresponding SimpleConfigurables are provided. Every next iteration
 * takes the result of previous iteration as a linearization point.
 * This is only needed in case of nonlinear constraints. In the
 * linear case result will be reached after the first step.
 * Stopping condition at the moment: sum of abs. constraint 
 * equations values at given point
 */

class LagrangeParentParticleFitter:public ParentParticleFitter
{

public:

 LagrangeParentParticleFitter();

 ~LagrangeParentParticleFitter(){}

/**
 * Refit method taking KinematicConstraint and vector
 * of trees as imput. Only top particles of corresponding trees are
 * refitted, vertex is not created. Number of trees should be one
 * (single track refit) or greater (multiple track refit). Some 
 * constraints may not work with single tracks (back to back for ex.)
 */ 
 vector<RefCountedKinematicTree>  fit(vector<RefCountedKinematicTree> trees, 
                                             KinematicConstraint * cs)const;
 
 LagrangeParentParticleFitter * clone() const
 {return new LagrangeParentParticleFitter(*this);}

private:

//tunable parameters and their read method
 void readParameters();
 float theMaxDiff;
 int theMaxStep;
};
#endif

#ifndef LagrangeParentParticleFitter_H
#define LagrangeParentParticleFitter_H

#include "RecoVertex/KinematicFit/interface/ParentParticleFitter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 * KinematicParticle refit via LMS minimization with Lagrange multipliers
 * method. Tunable for number of iterations and stopping condition.
 * Every new iteration
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
   * Configuration through PSet: number of iterations(maxDistance) and
   * stopping condition (maxNbrOfIterations)
   */

 void setParameters(const edm::ParameterSet& pSet);

/**
 * Refit method taking KinematicConstraint and vector
 * of trees as imput. Only top particles of corresponding trees are
 * refitted, vertex is not created. Number of trees should be one
 * (single track refit) or greater (multiple track refit). Some
 * constraints may not work with single tracks (back to back for ex.)
 */
 std::vector<RefCountedKinematicTree>  fit(const std::vector<RefCountedKinematicTree> & trees,
                                             KinematicConstraint * cs)const;

 LagrangeParentParticleFitter * clone() const
 {return new LagrangeParentParticleFitter(*this);}

private:

 void defaultParameters();

 float theMaxDiff;
 int theMaxStep;
};
#endif

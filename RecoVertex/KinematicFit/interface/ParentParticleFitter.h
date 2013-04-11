#ifndef ParentParticleFitter_H
#define ParentParticleFitter_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"

/**
 * Abstract Base class for mechanism
 * updating top tree particle with given 
 * constraint. To be used by KinematicParticleFitter
 * only.
 */

class ParentParticleFitter{

public:

  ParentParticleFitter(){}
  
 virtual ~ParentParticleFitter(){}
/**
 * Takes a kinematic tree as an input
 * The top tree particle get constrained
 */  
// virtual RefCountedKinematicTree  fit(RefCountedKinematicTree tree, KinematicConstraint * cs) const =0;
 
 
 virtual std::vector<RefCountedKinematicTree>  fit(const std::vector<RefCountedKinematicTree> &trees, 
                                                KinematicConstraint * cs) const =0;

/**
 * Clone method
 */
 virtual ParentParticleFitter * clone() const =0;
 
private:

};


#endif

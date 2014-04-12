#ifndef ChildUpdator_H
#define ChildUpdator_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"

/**
 * Abstract Base Class to update the 
 * daughter particles after the parent
 * particle was refitted. Implementation Returns the
 * refitted virtual particle with the pointers
 * to updated child particles
 *
 * Kirill Prokofiev, December 2002
 */

class ChildUpdator
{
public:
 
  ChildUpdator(){}
  
  virtual ~ChildUpdator(){}
/**
 * Method updating particles
 * and vertices inside the tree
 * below the constrained particle
 * The tree pointer should be set on
 * particle just updated by ParentParticleFitter
 * Class to be used by KinematicParticleFitter only.
 */  
  virtual RefCountedKinematicTree  update(RefCountedKinematicTree tree) const=0;
  
  virtual std::vector<RefCountedKinematicTree>  update(const std::vector<RefCountedKinematicTree> &trees) const=0;
  
  virtual ChildUpdator * clone() const = 0;

private:  

};
#endif

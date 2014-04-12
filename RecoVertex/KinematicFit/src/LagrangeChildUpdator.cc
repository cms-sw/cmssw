#include "RecoVertex/KinematicFit/interface/LagrangeChildUpdator.h"

RefCountedKinematicTree  LagrangeChildUpdator::update(RefCountedKinematicTree tree) const
{
//space for down update method
//now does nothing, supposed to
//update the states of daughter
//particles down the kinematic decay chain

 RefCountedKinematicTree nTree = tree;
 return nTree;
}
std::vector<RefCountedKinematicTree>  LagrangeChildUpdator::update(const std::vector<RefCountedKinematicTree> &trees) const
{
//space for down update method
//now does nothing, supposed to
//update the states of daughter
//particles down the kinematic decay chain

 std::vector<RefCountedKinematicTree> nTree = trees;
 
 return nTree;
}

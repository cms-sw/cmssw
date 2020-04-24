#ifndef LagrangeChildUpdator_H
#define LagrangeChildUpdator_H

#include "RecoVertex/KinematicFit/interface/ChildUpdator.h"

/**
 * This is  the space for daughter particle
 * update after the lagrange multipliers refit.
 * Current class is not yet implemented.
 * Return the input unchanged for the moment.
 */

class LagrangeChildUpdator:public ChildUpdator
{
public:

 LagrangeChildUpdator(){}
 ~LagrangeChildUpdator() override{}
 
 RefCountedKinematicTree  update(RefCountedKinematicTree tree) const override;
 
 std::vector<RefCountedKinematicTree>  update(const std::vector<RefCountedKinematicTree> & trees) const override;
 
 LagrangeChildUpdator * clone() const override
 {return new LagrangeChildUpdator(*this);}
 
private:

};
#endif

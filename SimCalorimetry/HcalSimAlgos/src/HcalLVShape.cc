#include "SimCalorimetry/HcalSimAlgos/interface/HcalLVShape.h"
  
HcalLVShape::HcalLVShape()
: shape_(HcalPulseShapes().hbShape(true))
{
}

double HcalLVShape::timeToRise() const 
{
   return 0.;
}

double HcalLVShape::operator () (double time_) const
{
  return shape_.at(time_);
}



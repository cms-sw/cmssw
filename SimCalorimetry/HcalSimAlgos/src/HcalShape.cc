#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
  
HcalShape::HcalShape()
: shape_(HcalPulseShapes().hbShape())
{
}

double HcalShape::timeToRise() const 
{
   return 0.;
}

double HcalShape::operator () (double time_) const
{
  return shape_.at(time_);
}



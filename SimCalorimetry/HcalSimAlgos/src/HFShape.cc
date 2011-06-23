#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
  
HFShape::HFShape()
: shape_(HcalPulseShapes().hfShape())
{   
}

double
HFShape::timeToRise() const 
{
   return 0. ;
}
  
double HFShape::operator () (double time) const
{
  return shape_.at(time);
}
  

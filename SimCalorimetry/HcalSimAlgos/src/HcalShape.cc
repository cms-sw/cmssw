#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
  
HcalShape::HcalShape()
// : shape_(HcalPulseShapes().hbShape())
{
   // no more defual shape is defined (since cmssw 5x)
}

void HcalShape::setShape(int shapeType)
{
   // keep pulse shape for HPD, HO SiPM, HF PMT, depending on shapeType 
  // (101,102 etc.)
 //  std::cout << "- HcalShape::setShape for type " << shapeType << std::endl;
   shape_=HcalPulseShapes().getShape(shapeType);
}

double HcalShape::timeToRise() const 
{
   return 0.;
}

double HcalShape::operator () (double time_) const
{
  return shape_.at(time_);
}



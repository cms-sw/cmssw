#ifndef HcalSimAlgos_HcalLVShape_h
#define HcalSimAlgos_HcalLVShape_h
#include<vector>
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
/**

   \class HcalShape

   \brief  shaper for Hcal (not for HF)
   
*/

class HcalLVShape : public CaloVShape
{
public:
  HcalLVShape();
  virtual double operator () (double time) const;
  virtual double timeToRise() const;
private:
  HcalPulseShapes::Shape shape_;

};

#endif
  
  

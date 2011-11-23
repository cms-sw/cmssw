#ifndef HcalSimAlgos_HcalShape_h
#define HcalSimAlgos_HcalShape_h
#include<vector>
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
/**

   \class HcalShape

   \brief  shaper for Hcal (not for HF)
   
*/

class HcalShape : public CaloVShape
{
public:
  HcalShape();
  void setShape(int shapeType);
  virtual double operator () (double time) const;
  virtual double timeToRise() const;
private:
  HcalPulseShapes::Shape shape_;

};

#endif
  
  

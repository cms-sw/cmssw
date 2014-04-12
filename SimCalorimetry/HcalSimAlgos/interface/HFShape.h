#ifndef HcalSimAlgos_HFShape_h
#define HcalSimAlgos_HFShape_h
#include<vector>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
/**
   \class HFShape
  
   \brief  shaper for HF
     
*/
class HFShape : public CaloVShape
{
public:
  HFShape();
  virtual ~HFShape(){}
  
  virtual double operator () (double time) const;
  virtual double timeToRise() const;

 private:
   HcalPulseShapes::Shape shape_; 
};

#endif
  

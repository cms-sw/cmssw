#ifndef HcalSimAlgos_HcalShape_h
#define HcalSimAlgos_HcalShape_h
#include <vector>

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
/**

   \class HcalShape

   \brief  shaper for Hcal (not for HF)
   
*/

class HcalShape : public CaloVShape {
public:
  HcalShape();
  void setShape(int shapeType);
  double operator()(double time) const override;
  double timeToRise() const override;

private:
  HcalPulseShapes::Shape shape_;
};

#endif

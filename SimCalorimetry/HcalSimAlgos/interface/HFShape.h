#ifndef HcalSimAlgos_HFShape_h
#define HcalSimAlgos_HFShape_h
#include <vector>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
/**
   \class HFShape
  
   \brief  shaper for HF
     
*/
class HFShape : public CaloVShape {
public:
  HFShape();
  ~HFShape() override {}

  double operator()(double time) const override;
  double timeToRise() const override;

private:
  HcalPulseShapes::Shape shape_;
};

#endif

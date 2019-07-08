#ifndef HcalSimAlgos_ZDCShape_h
#define HcalSimAlgos_ZDCShape_h
#include <vector>

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

/**
  
   \class ZDCShape
  
   \brief  shaper for ZDC
     
*/

class ZDCShape : public CaloVShape {
public:
  ZDCShape();
  ZDCShape(const ZDCShape& d);

  ~ZDCShape() override {}

  double operator()(double time) const override;
  double timeToRise() const override;

private:
  void computeShapeZDC();

  int nbin_;
  std::vector<float> nt_;
};

#endif

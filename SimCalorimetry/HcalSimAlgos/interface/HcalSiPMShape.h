// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPMShape_h
#define HcalSimAlgos_HcalSiPMShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include <vector>

class HcalSiPMShape : public CaloVShape {
public:
  HcalSiPMShape(unsigned int signalShape = 206);
  HcalSiPMShape(const HcalSiPMShape& other);

  ~HcalSiPMShape() override {}

  double operator()(double time) const override;

  double timeToRise() const override { return 0.0; }

protected:
  void computeShape(unsigned int signalShape);

private:
  int nBins_;
  std::vector<double> nt_;
};

#endif  //HcalSimAlgos_HcalSiPMShape_h

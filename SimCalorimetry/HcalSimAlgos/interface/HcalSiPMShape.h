// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPMShape_h
#define HcalSimAlgos_HcalSiPMShape_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include <vector>

class HcalSiPMShape final : public CaloVShape {
public:
  HcalSiPMShape(unsigned int signalShape = 206);
  HcalSiPMShape(const HcalSiPMShape& other);

  ~HcalSiPMShape() override {}

  int nBins() const { return nBins_; }
  double operator[](int i) const { return nt_[i]; }

  double operator()(double time) const override {
    int jtime(time * HcalPulseShapes::invDeltaTSiPM_ + 0.5);
    return (jtime >= 0 && jtime < nBins_) ? nt_[jtime] : 0;
  }

  double timeToRise() const override { return 0.0; }

protected:
  void computeShape(unsigned int signalShape);

private:
  int nBins_;
  std::vector<double> nt_;
};

#endif  //HcalSimAlgos_HcalSiPMShape_h

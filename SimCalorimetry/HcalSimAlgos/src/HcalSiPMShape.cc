#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include <iostream>

HcalSiPMShape::HcalSiPMShape(unsigned int signalShape)
    : CaloVShape(), nBins_(HcalPulseShapes::nBinsSiPM_ * HcalPulseShapes::invDeltaTSiPM_), nt_(nBins_, 0.) {
  computeShape(signalShape);
}

HcalSiPMShape::HcalSiPMShape(const HcalSiPMShape& other) : CaloVShape(other), nBins_(other.nBins_), nt_(other.nt_) {}

void HcalSiPMShape::computeShape(unsigned int signalShape) {
  //grab correct function pointer based on shape
  double (*analyticPulseShape)(double);
  if (signalShape == HcalShapes::ZECOTEK || signalShape == HcalShapes::HAMAMATSU)
    analyticPulseShape = &HcalPulseShapes::analyticPulseShapeSiPMHO;
  else if (signalShape == HcalShapes::HE2017 or signalShape == HcalShapes::HE2018)
    analyticPulseShape = &HcalPulseShapes::analyticPulseShapeSiPMHE;
  else
    return;

  double norm = 0.;
  for (int j = 0; j < nBins_; ++j) {
    nt_[j] = analyticPulseShape(j * HcalPulseShapes::deltaTSiPM_);
    norm += (nt_[j] > 0) ? nt_[j] : 0.;
  }

  for (int j = 0; j < nBins_; ++j) {
    nt_[j] /= norm;
  }
}

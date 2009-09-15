#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"

#include <cmath>

HcalSiPMShape::HcalSiPMShape() : CaloVShape(), nBins_(512), nt_(nBins_, 0.) {
  computeShape();
}

HcalSiPMShape::HcalSiPMShape(const HcalSiPMShape & other) :
  CaloVShape(other), nBins_(other.nBins_), nt_(other.nt_) {
}

double HcalSiPMShape::operator () (double time) const {
  int jtime = static_cast<int>(time + 0.5);
  if (jtime>=0 && jtime<nBins_) 
    return nt_[jtime];
  return 0.;
}

void HcalSiPMShape::computeShape() {

  double norm = 0.;
  for (int j = 0; j < nBins_; ++j) {
    if (j <= 31.)
      nt_[j] = 2.15*j;
    else if ((j > 31) && (j <= 96))
      nt_[j] = 102.1 - 1.12*j;
    else 
      nt_[j] = 0.0076*j - 6.4;
    norm += (nt_[j]>0) ? nt_[j] : 0.;
  }

  for (int j = 0; j < nBins_; ++j) {
    nt_[j] /= norm;
  }
}

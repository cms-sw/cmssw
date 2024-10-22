#include "SimCalorimetry/CaloSimAlgos/interface/CaloCachedShapeIntegrator.h"

const int NBINS = 281;  // 256, plus 25 before

CaloCachedShapeIntegrator::CaloCachedShapeIntegrator(const CaloVShape *aShape)
    : v_(NBINS, 0.), timeToRise_(aShape->timeToRise()) {
  for (int t = 0; t < 256; ++t) {
    double amount = (*aShape)(t);
    for (int istep = 0; istep < 25; ++istep) {
      int ibin = t + istep;
      v_[ibin] += amount;
    }
  }
}

CaloCachedShapeIntegrator::~CaloCachedShapeIntegrator() {}

double CaloCachedShapeIntegrator::timeToRise() const { return timeToRise_; }

double CaloCachedShapeIntegrator::operator()(double startTime) const {
  // round up, and account for the -25 ns offset
  int ibin = static_cast<int>(startTime + 25.0);
  return (ibin < 0 || ibin >= NBINS) ? 0. : v_[ibin];
}

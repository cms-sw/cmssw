#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CastorSim/interface/CastorShape.h"
#include <cmath>

CastorShape::CastorShape() : nbin_(256), nt_(nbin_, 0.) { computeShapeCastor(); }

CastorShape::CastorShape(const CastorShape &d) : CaloVShape(d), nbin_(d.nbin_), nt_(d.nt_) {}

void CastorShape::computeShapeCastor() {
  edm::LogVerbatim("CastorSim") << "\n ===== computeShapeCastor  !!! \n";

  const float ts = 3.0;  // time constant in   t * exp(-(t/ts)**2)

  int j;
  float norm;

  // HF SHAPE
  norm = 0.0;
  for (j = 0; j < 3 * ts && j < nbin_; j++) {
    // nt_[j] = ((float)j)*exp(-((float)(j*j))/(ts*ts));
    nt_[j] = j * exp(-(j * j) / (ts * ts));
    norm += nt_[j];
  }
  // normalize pulse area to 1.0
  for (j = 0; j < 3 * ts && j < nbin_; j++) {
    nt_[j] /= norm;
  }
}

double CastorShape::operator()(double time) const {
  // return pulse amplitude for request time in ns
  int jtime;
  jtime = static_cast<int>(time + 0.5);

  if (jtime >= 0 && jtime < nbin_)
    return nt_[jtime];
  else
    return 0.0;
}

double CastorShape::timeToRise() const { return 0.0; }

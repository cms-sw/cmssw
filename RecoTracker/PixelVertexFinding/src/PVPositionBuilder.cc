#include "RecoTracker/PixelVertexFinding/interface/PVPositionBuilder.h"
#include <cmath>

/// Constructor does nothing since this class has no data
PVPositionBuilder::PVPositionBuilder() {}

Measurement1D PVPositionBuilder::average(const reco::TrackRefVector &trks) const {
  // Cut and paste (more or less) from same class in ORCA framework
  double ntracks = double(trks.size());
  if (ntracks == 0)
    return Measurement1D(0., 0.);
  double sumZIP = 0;
  double err = 0;
  for (unsigned int i = 0; i < trks.size(); i++) {
    sumZIP += trks[i]->dz();    // Z at IP
    err += trks[i]->dzError();  // error on Z at IP (I hope). Fix d.k.
  }
  return Measurement1D(sumZIP / ntracks, err / ntracks / std::sqrt(ntracks));
}

Measurement1D PVPositionBuilder::wtAverage(const reco::TrackRefVector &trks) const {
  double ntracks = double(trks.size());
  if (ntracks == 0)
    return Measurement1D(0.0, 0.0);
  double sumUp = 0;
  double sumDown = 0;
  double err = 0;
  for (unsigned int i = 0; i < trks.size(); i++) {
    //    double err2 = trks[i]->covariance(3,3); // error on Z at IP (I hope)
    double err2 = trks[i]->dzError();  // Fix d.k.
    err2 *= err2;
    if (err2 != 0) {
      sumUp += trks[i]->dz() * 1 / err2;  // error-weighted average of Z at IP
      sumDown += 1 / err2;
    }
    err += std::sqrt(err2);
  }
  if (sumDown > 0)
    return Measurement1D(sumUp / sumDown, err / ntracks / sqrt(ntracks));
  else
    return Measurement1D(0.0, 0.0);
}

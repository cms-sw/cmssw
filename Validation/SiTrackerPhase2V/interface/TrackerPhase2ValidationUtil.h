
#ifndef _Validation_SiTrackerPhase2V_TrackerPhase2ValidationUtil_h
#define _Validation_SiTrackerPhase2V_TrackerPhase2ValidationUtil_h

#include <cmath>
#include <tuple>  // for std::tuple

#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

namespace phase2tkutil {

  bool isPrimary(const SimTrack& simTrk, const PSimHit* simHit);
  static constexpr float cmtomicron = 1e4;

  inline std::tuple<float, float, float> computeZ0LxyD0(const TrackingParticle& tp) {
    trklet::Settings settings;
    double bfield_ = settings.bfield();
    double c_ = settings.c();

    float vz = tp.vz();
    float vx = tp.vx();
    float vy = tp.vy();
    float charge = tp.charge();
    float pt = tp.pt();
    float eta = tp.eta();
    float phi = tp.phi();

    float t = std::tan(2.0 * std::atan(1.0) - 2.0 * std::atan(std::exp(-eta)));
    float K = 0.01 * bfield_ * c_ / 2 / pt * charge;  // curvature correction
    float A = 1. / (2. * K);
    float x0p = -vx - A * std::sin(phi);
    float y0p = -vy + A * std::cos(phi);
    float delphi = reco::deltaPhi(phi, std::atan2(-K * x0p, K * y0p));

    float z0 = vz + t * delphi / (2.0 * K);
    float Lxy = std::sqrt(vx * vx + vy * vy);
    float rp = std::sqrt(x0p * x0p + y0p * y0p);
    float d0 = charge * rp - (1. / (2. * K));

    float alt_d0 = -vx * std::sin(phi) + vy * std::cos(phi);
    d0 = -d0;  // Fix d0 sign

    if (K == 0) {
      d0 = alt_d0;
      z0 = vz;
    }

    return std::make_tuple(z0, Lxy, d0);
  }

}  // namespace phase2tkutil

#endif

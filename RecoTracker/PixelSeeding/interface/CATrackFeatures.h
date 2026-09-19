#ifndef RecoTracker_PixelSeeding_interface_CATrackFeatures_h
#define RecoTracker_PixelSeeding_interface_CATrackFeatures_h

// Feature vector of the displaced-track classifier (CATrackDNN.h), in the trained order. fill() is
// shared by Kernel_classifyTracks (Stage 1, the 12 features), PixelTrackTorchHighPuritySelector
// (Stage 2, plus the rzKappaOut extras) and the host table producer, which must agree bit for bit.

#include <cmath>
#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"

namespace caTrackFeatures {

  // isStub is resolved by ADL on the hit view (caStructures::isStub for CAHitsView); keep the call
  // unqualified.

  inline constexpr int kNFeat = 12;

  // Inverse-variance stub-curvature kernel: given rg2, bend d=dPhiDr and error s=dPhiDrError, return
  // den = 1 + rg2*d^2 and weight w = den^3 / s^2. rg2 is caller-passed because the xg^2+yg^2 and
  // rGlobal()^2 forms differ at the bit level and each call site must stay exact for its model.
  ALPAKA_FN_HOST_ACC inline void stubDenWeight(float rg2, float d, float s, float &den, float &w) {
    den = 1.f + rg2 * d * d;
    w = den * den * den / (s * s);
  }

  // Feature order, as trained:
  //   0 fitChi2  1 psFrac  2 r0  3 nPS  4 nh  5 spanZ
  //   6 nStubs   7 nl      8 logChi2Stub  9 kErr  10 dcaEst  11 nBarrel
  // HitIter: forward iterator over hit indices; returns false on an empty/corrupt list.
  // otView: raw OT-rechit SoA for hit ids with kOTHitTag set (otIdx(id)); such a hit is not a stub and
  // rGlobal() is derived inline. A null view with tagged ids -> false.
  // rzKappaOut (optional, 4 floats): out[0]=rzChi2 (straight line z=a+b*r; -1 undefined),
  // out[1]=meanStubKappa, out[2]=leverArm (rMax-r0), out[3]=rMax.
  // HitsView: the pixel+stubs CAHitsView facade (see CAHitsView.h), on the host as in the kernels.
  template <typename HitIter, typename HitsView>
  ALPAKA_FN_HOST_ACC inline bool fill(HitIter hitBegin,
                                      HitIter hitEnd,
                                      HitsView const &hh,
                                      int nHitsTot,
                                      float nLayers,
                                      float chi2,
                                      float *feat,
                                      float *rzKappaOut = nullptr,
                                      ::reco::OTRecHitsConstView const *otView = nullptr) {
    int nh = 0;
    for (auto ph = hitBegin; ph != hitEnd; ++ph)
      ++nh;
    if (nh < 3)
      return false;

    float x0 = 0.f, y0 = 0.f, z0 = 0.f, r0 = 0.f;
    float xm = 0.f, ym = 0.f;
    float xN = 0.f, yN = 0.f, zN = 0.f;
    int nStubs = 0, nPS = 0, nBarrel = 0, nStubK = 0;
    float sumW = 0.f, sumWK = 0.f, sumWK2 = 0.f;
    // Stage-2 extras accumulators, touched only when rzKappaOut != nullptr.
    float sumKw = 0.f, sumKwk = 0.f;
    float rMaxE = 0.f, r0E = -1.f;  // radial extent, rGlobal-based
    double Sr = 0, Sz = 0, Srr = 0, Srz = 0, Szz = 0;
    const int iMid = nh / 2;
    int i = 0;
    for (auto ph = hitBegin; ph != hitEnd; ++ph, ++i) {
      const uint32_t h = *ph;
      float xg, yg, zg;
      bool otHit = false;
      if (caOTHitTag::isOTId(h)) {
        // Raw OT-rechit extra: the id indexes the OT SoA, not hh. Without otView -> bail.
        if (!otView)
          return false;
        const uint32_t o = caOTHitTag::otIdx(h);
        // Corruption guard: a container-content overflow can produce garbage words with the tag bit set.
        if (o >= uint32_t(otView->metadata().size()))
          return false;
        xg = (*otView)[o].xGlobal();
        yg = (*otView)[o].yGlobal();
        zg = (*otView)[o].zGlobal();
        otHit = true;
      } else {
        if (h >= static_cast<uint32_t>(nHitsTot))
          return false;  // content overflow corruption guard
        xg = hh[h].xGlobal();
        yg = hh[h].yGlobal();
        zg = hh[h].zGlobal();
      }
      if (i == 0) {
        x0 = xg, y0 = yg, z0 = zg, r0 = std::sqrt(xg * xg + yg * yg);
      }
      if (i == iMid)
        xm = xg, ym = yg;
      xN = xg, yN = yg, zN = zg;
      // r-z linearity sums, rGlobal()-based (the OT SoA derives rGlobal from x/y).
      float rg = 0.f;
      if (rzKappaOut) {
        rg = otHit ? std::sqrt(xg * xg + yg * yg) : hh[h].rGlobal();
        if (r0E < 0.f)
          r0E = rg;  // first hit's radius
        rMaxE = std::max(rMaxE, rg);
        Sr += rg;
        Sz += zg;
        Srr += double(rg) * rg;
        Srz += double(rg) * zg;
        Szz += double(zg) * zg;
      }
      // OT extras are never stubs, so they contribute as a pixel hit does.
      if (!otHit && isStub(hh, h)) {
        ++nStubs;
        auto const stub = hh.stub(int32_t(h));
        if (stub.isPS())
          ++nPS;
        if (stub.isBarrel())
          ++nBarrel;
        const float s = stub.dPhiDrError();
        if (s > 0.f) {
          const float d = stub.dPhiDr();
          float den, w;
          stubDenWeight(xg * xg + yg * yg, d, s, den, w);
          const float k = d / std::sqrt(den);
          sumW += w;
          sumWK += w * k;
          sumWK2 += w * k * k;
          ++nStubK;
          if (rzKappaOut) {
            // meanStubKappa weight uses rGlobal()^2, not xg^2+yg^2, to match the host table bit for bit.
            float denR, wR;
            stubDenWeight(rg * rg, d, s, denR, wR);
            sumKw += wR;
            sumKwk += wR * (d / std::sqrt(denR));
          }
        }
      }
    }

    const float kErr = (sumW > 0.f) ? 1.f / std::sqrt(sumW) : -1.f;
    const float chi2Stub = (nStubK >= 3 && sumW > 0.f) ? (sumWK2 - sumWK * sumWK / sumW) / float(nStubK - 1) : -1.f;
    CircleEq<float> eq(x0, y0, xm, ym, xN, yN);
    const float curv3 = eq.curvature();
    const float dcaEst = (std::abs(curv3) > 0.f) ? std::abs(eq.dca0()) / std::abs(curv3) : 0.f;

    // fitChi2: non-finite, negative or huge raw chi2 -> worst cap, then log1p, as the trainer does.
    float fitChi2Safe = chi2;
    if (!(fitChi2Safe >= 0.f) || fitChi2Safe > 1.0e4f)  // catches NaN, negative, +inf, huge
      fitChi2Safe = 1.0e4f;
    feat[0] = std::log1p(fitChi2Safe);
    feat[1] = float(nPS) / float(std::max(nStubs, 1));
    feat[2] = r0;
    feat[3] = float(nPS);
    feat[4] = float(nh);
    feat[5] = std::abs(zN - z0);
    feat[6] = float(nStubs);
    feat[7] = nLayers;
    feat[8] = std::log1p(std::max(chi2Stub, 0.f));
    feat[9] = kErr;
    feat[10] = dcaEst;
    feat[11] = float(nBarrel);
    // Stage-2 extras, finalized from the walk above; same arithmetic as the host producer.
    if (rzKappaOut) {
      rzKappaOut[1] = (sumKw > 0.f) ? sumKwk / sumKw : 0.f;  // meanStubKappa
      float rz = -1.f;                                       // rzChi2 (-1 = undefined)
      // The nh hits were all validated above.
      if (nh >= 3) {
        const double D = nh * Srr - Sr * Sr;
        if (std::abs(D) > 0.0) {
          const double b = (nh * Srz - Sr * Sz) / D, a = (Sz - b * Sr) / nh;
          rz = float(std::max(0.0, (Szz - a * Sz - b * Srz)) / std::max(1, nh - 2));
        }
      }
      rzKappaOut[0] = rz;
      // Radial extent, as in CATrackFeaturesTableProducer.
      rzKappaOut[2] = rMaxE - (r0E < 0.f ? 0.f : r0E);  // leverArm
      rzKappaOut[3] = rMaxE;                            // rMax
    }
    return true;
  }

}  // namespace caTrackFeatures

#endif  // RecoTracker_PixelSeeding_interface_CATrackFeatures_h

#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNN_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNN_h

// Inline MLP with compile-time weights, evaluated per-triplet inside Kernel_connect (see CATripletCuts.h
// accept(), gated by the top-level useTripletDNN producer flag). Forward pass:
// standardization (kMean/kScale) -> relu -> relu -> sigmoid. The weights live in an
// auto-generated header (CATripletDNNWeights_prompt.h, baked by
// RecoTracker/PixelSeeding/test/models/retrain_triplet_gate.sh) and are indexed directly as
// constexpr globals, so the whole forward pass inlines (register/stack neutral). The in-kernel
// feature vector is filled in the SAME order (and with the same derived-feature formulas) as
// that script's BASE_FEATURES + DERIVED lists.

#include <alpaka/alpaka.hpp>
#include <array>
#include <cmath>
#include <type_traits>

#include "CATripletDNNWeights.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The gate's input, 18 raw and 11 derived quantities; asArray() gives them in the trained order.
  struct CATripletFeatures {
    float absCurvature = 0.f;
    float tipTimesCurvature = 0.f;
    float dca = 0.f;
    float curvatureStubs = 0.f;
    float curvatureStubsErrSquared = 0.f;
    float curvature13 = 0.f;
    float dPhi12 = 0.f;
    float dPhi13 = 0.f;
    float dPhi23 = 0.f;
    float dr12 = 0.f;
    float dr13 = 0.f;
    float r1 = 0.f;
    float r2 = 0.f;
    float r3 = 0.f;
    float z1 = 0.f;
    float z2 = 0.f;
    float z3 = 0.f;
    float nStubs = 0.f;
    float stubCirclePull = 0.f;
    float stubCircleRatio = 0.f;
    float curv13Resid = 0.f;
    float rzResid = 0.f;
    float cotTheta = 0.f;
    float dPhiRatio = 0.f;
    float logAbsCurv = 0.f;
    float logErrSq = 0.f;
    float logDca = 0.f;
    float layGap12 = 0.f;
    float layGap23 = 0.f;

    ALPAKA_FN_HOST_ACC constexpr std::array<float, caTripletDNN::kNFeat> asArray() const {
      return {absCurvature,
              tipTimesCurvature,
              dca,
              curvatureStubs,
              curvatureStubsErrSquared,
              curvature13,
              dPhi12,
              dPhi13,
              dPhi23,
              dr12,
              dr13,
              r1,
              r2,
              r3,
              z1,
              z2,
              z3,
              nStubs,
              stubCirclePull,
              stubCircleRatio,
              curv13Resid,
              rzResid,
              cotTheta,
              dPhiRatio,
              logAbsCurv,
              logErrSq,
              logDca,
              layGap12,
              layGap23};
    }
  };
  static_assert(std::is_standard_layout_v<CATripletFeatures>);

  namespace caTripletDNN_eval {

    // feat[] has kNFeat entries in the trained order. Returns P(real) in [0,1].
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float score(const float* feat) {
      using namespace caTripletDNN_prompt;
      float xn[kNFeat];
      for (int i = 0; i < kNFeat; ++i)
        xn[i] = (feat[i] - kMean[i]) / kScale[i];
      float h1[kNH1];
      for (int j = 0; j < kNH1; ++j) {
        float s = kB1[j];
        for (int i = 0; i < kNFeat; ++i)
          s += xn[i] * kW1[i * kNH1 + j];  // kW1 row-major [kNFeat][kNH1]
        h1[j] = s > 0.f ? s : 0.f;         // relu
      }
      float h2[kNH2];
      for (int k = 0; k < kNH2; ++k) {
        float s = kB2[k];
        for (int j = 0; j < kNH1; ++j)
          s += h1[j] * kW2[j * kNH2 + k];  // kW2 row-major [kNH1][kNH2]
        h2[k] = s > 0.f ? s : 0.f;         // relu
      }
      float o = kB3[0];
      for (int k = 0; k < kNH2; ++k)
        o += h2[k] * kW3[k];              // kW3 [kNH2][1]
      return 1.f / (1.f + std::exp(-o));  // sigmoid
    }

  }  // namespace caTripletDNN_eval
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

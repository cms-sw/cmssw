#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletCuts_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletCuts_h

#include <alpaka/alpaka.hpp>
#include <cmath>

#include "FWCore/Utilities/interface/isFinite.h"  // bit-pattern finiteness test for the DNN-gate inputs
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/CAStubMS.h"
#include "CACell.h"
#include "CAPipelineCounters.h"
#include "CATripletDNN.h"  // inline per-triplet DNN gate (compile-time weights)

// CA_TRIPLET_DUMP (built-triplet dataset dump, the truth-labeled DNN training input) is toggled in
// this minimal header so the producer side can see it without pulling in this device header. It
// must stay commented out in production.
#include "CATripletDumpMacro.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  struct TripletCuts {
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    // ----------------------------------
    // RZ alignment cut aka CAThetaCut
    // ----------------------------------
    // This cut checks the alignment of the three hits in the RZ plane by applying a cut on the angle between the middle
    // and outer hit with respect to the inner hit.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool alignedRZ(const float r1,
                                                         const float z1,
                                                         const float r2,
                                                         const float z2,
                                                         const float r3,
                                                         const float z3,
                                                         const float ptmin,
                                                         const float maxRZTolerance) {
      float dr13 = std::abs(r1 - r3);
      float dist13Squared = dr13 * dr13 + (z1 - z3) * (z1 - z3);

      float pMin = ptmin * std::sqrt(dist13Squared);  // this needs to be divided by dr13 later

      float tan_12_13_half_mul_dist13Squared = fabs(z1 * (r2 - r3) + z2 * (r3 - r1) + z3 * (r1 - r2));
      bool aligned = tan_12_13_half_mul_dist13Squared * pMin <= maxRZTolerance * dist13Squared * dr13;

#ifdef CA_DEBUG
      float lhs = tan_12_13_half_mul_dist13Squared * pMin;
      float rhs = maxRZTolerance * dist13Squared * dr13;
      printf(
          "TripletCuts::alignedRZ;r1=%.4f;z1=%.4f;r2=%.4f;z2=%.4f;r3=%.4f;z3=%.4f;lhs=%.6f;rhs=%.6f;maxRZTolerance=%."
          "6f;"
          "ptmin=%.6f;aligned=%d\n",
          r1,
          z1,
          r2,
          z2,
          r3,
          z3,                // hit positions (r,z for 3 hits)
          lhs,               // alignment value (LHS of check)
          rhs,               // threshold (RHS of check)
          maxRZTolerance,    // raw maxRZTolerance parameter
          ptmin,             // ptmin parameter
          aligned ? 1 : 0);  // pass/fail
#endif

      return aligned;
    }

    // ---------------------------------
    // XY alignment cut aka hardCurvCut
    // ---------------------------------
    // This is a simple cut on the curvature computed from the three hits in the transverse plane.
    // It is indirectly setting a minimum pT cut for triplets and checks their alignment in the transverse plane.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool alignedXY(const float absCurvature, const float maxCurvature) {
      bool aligned = absCurvature <= maxCurvature;

#ifdef CA_DEBUG
      printf("TripletCuts::alignedXY;absCurvature=%.4f;maxCurvature=%.4f;aligned=%d\n",
             absCurvature,      // computed absolute curvature
             maxCurvature,      // curvature threshold
             aligned ? 1 : 0);  // pass/fail
#endif

      return aligned;
    }

    // ---------------------------------
    // check on compatibility with the beamspot in XY via Transverse Impact Parameter aka caDCACut
    // ---------------------------------
    // This cut checks the compatibility of the triplet with the beamspot in the transverse plane by applying
    // a cut on the transverse impact parameter (DCA) computed from the three hits.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool beamspotCompatibleXY(const float absCurvature,
                                                                    const float tipTimesCurvature,
                                                                    const float maxTip,
                                                                    const float floorTipTimesCurvature = 0.f) {
      // |dca0| < maxTip * |curvature| (upstream's CACell::dcaCut), plus an additive floor that is
      // inert unless configured: the geometry builder writes -1 for an unset floorDCA and
      // std::max(-1.f, 0.f) == 0.
      float maxTipTimesCurvature = maxTip * absCurvature + std::max(floorTipTimesCurvature, 0.f);
      bool compatible = tipTimesCurvature < maxTipTimesCurvature;

#ifdef CA_DEBUG
      printf(
          "TripletCuts::beamspotCompatibleXY;absCurvature=%.4f;maxTip=%.4f;tipTimesCurvature=%.4f;"
          "floorTipTimesCurvature=%.4f;maxTipTimesCurvature=%.4f;compatible=%d\n",
          absCurvature,            // computed absolute curvature
          maxTip,                  // maximum tip distance
          tipTimesCurvature,       // tip times curvature
          floorTipTimesCurvature,  // floor value for tip times curvature
          maxTipTimesCurvature,    // maximum tip times curvature
          compatible ? 1 : 0);     // pass/fail
#endif

      return compatible;
    }

    // ---------------------------------
    // dPhi same sign cut
    // ---------------------------------
    // Requires dPhi12 and dPhi23 to have the same sign. A track that bends in one direction and
    // originates near the beam line steps in phi monotonically from layer to layer; tracks with a
    // large impact parameter or strong multiple scattering can violate this, so the cut is optional
    // (sameDPhiSign, on for the stub topology where the triplet phi lever arm is long).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool sameSignDPhi(const float dPhi12, const float dPhi23) {
      bool sameSign = (dPhi12 * dPhi23) >= 0.f;

#ifdef CA_DEBUG
      printf("TripletCuts::sameSignDPhi;dPhi12=%.4f;dPhi23=%.4f;sameSign=%d\n",
             dPhi12,             // dPhi between inner and middle hit
             dPhi23,             // dPhi between middle and outer hit
             sameSign ? 1 : 0);  // pass/fail
#endif

      return sameSign;
    }

    // ---------------------------------
    // curvature compatibility with inner doublet aka caPhiMiddleCut
    // ---------------------------------
    // This cut checks the compatibility of the inner doublet's dPhi/dr with the stubs' average of the triplet.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool stubsCompatibleWithInnerDoublet(const float dPhi12,
                                                                               const float dr12,
                                                                               const float curvatureStubs,
                                                                               const float maxStubInnerDoubletDCurv) {
      if (maxStubInnerDoubletDCurv < 0.f)
        return true;  // cut disabled

      float dPhiDiff = std::abs(dPhi12 - curvatureStubs * dr12);
      bool compatible = dPhiDiff < maxStubInnerDoubletDCurv * dr12;

#ifdef CA_DEBUG
      printf(
          "TripletCuts::stubsCompatibleWithInnerDoublet;dPhi12=%.4f;dr12=%.4f;curvatureStubs=%.4f;"
          "maxStubInnerDoubletDCurv=%.4f;"
          "dPhiDiff=%.4f;compatible=%d\n",
          dPhi12,                    // dPhi between inner and middle hit
          dr12,                      // dr between inner and middle hit
          curvatureStubs,            // curvature from stubs
          maxStubInnerDoubletDCurv,  // maximum allowed dPhi/dr difference
          dPhiDiff,                  // computed dPhi/dr difference
          compatible ? 1 : 0);       // pass/fail
#endif

      return compatible;
    }

    // Tangent of the three-point circle at one stub, as a half-curvature. The circle through the triplet's
    // three hits knows the impact parameter, so its tangent at the stub's radius already contains the d0/r^2
    // term the stub measures and the comparison is free of d0. cosdir(x,y) is the unit radial vector of the
    // circle at (x,y) up to the sign of the curvature; sin(beta)/r with beta the angle to the radial direction
    // is the same quantity as the stub's own d/sqrt(1 + r^2 d^2).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static float circleKappaAt(CircleEq<float> const& eq, float x, float y) {
      auto cd = eq.cosdir(x, y);
      float tx = cd.second, ty = -cd.first;
      if (tx * x + ty * y < 0.f) {
        tx = -tx;
        ty = -ty;
      }
      float r2 = x * x + y * y;
      return (r2 > 0.f) ? (x * ty - y * tx) / r2 : 0.f;
    }

    // curvature compatibility with the triplet aka geomKappaSigmaCut
    // Inverse-variance weighted mean of the per-stub residuals against the three-point circle's own
    // tangent, in sigma. Errors from the precision-only bend column: the plain one carries the
    // along-strip term twice although it cancels in the bend, which on endcap 2S stubs makes the
    // cut about twenty times looser than the hit precision warrants.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool stubsCurvCompatibleWithTriplet(const float residSum,
                                                                              const float weightSum,
                                                                              const float maxStubGeomCurvSigma) {
      if (maxStubGeomCurvSigma < 0.f || !(weightSum > 0.f))
        return true;  // cut disabled, or no stub with a usable error

      float resid = residSum / weightSum;
      bool compatible = resid * resid * weightSum < maxStubGeomCurvSigma * maxStubGeomCurvSigma;

#ifdef CA_DEBUG
      printf(
          "TripletCuts::stubsCurvCompatibleWithTriplet;resid=%.6f;sigma=%.6f;maxStubGeomCurvSigma=%.4f;compatible=%d\n",
          resid,
          std::sqrt(1.f / weightSum),
          maxStubGeomCurvSigma,
          compatible ? 1 : 0);
#endif

      return compatible;
    }

    // ---------------------------------
    // phi compatibility
    // ---------------------------------
    // This cut checks the phi compatibility of the three hits by comparing the phi of the middle hit with the phi
    // predicted from the inner and outer hit.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool phiCompatible(
        const float dPhi12, const float dr12, const float dPhi13, const float dr13, const float maxPhiResid) {
      if (maxPhiResid < 0.f)
        return true;  // cut disabled

      float phiResid = std::abs(dPhi12 * dr13 - dPhi13 * dr12);
      bool compatible = phiResid < maxPhiResid * dr13;

#ifdef CA_DEBUG
      printf(
          "TripletCuts::phiCompatible;dPhi12=%.4f;dr12=%.4f;dPhi13=%.4f;dr13=%.4f;maxPhiResid=%.4f;phiResid=%.4f;"
          "compatible=%d\n",
          dPhi12,               // dPhi between inner and middle hit
          dr12,                 // dr between inner and middle hit
          dPhi13,               // dPhi between inner and outer hit
          dr13,                 // dr between inner and outer hit
          maxPhiResid,          // maximum allowed phi residual
          phiResid,             // computed phi residual
          compatible ? 1 : 0);  // pass/fail
#endif

      return compatible;
    }

    // -------------------------------------------------------------------------------------------------------------
    // MAIN FUNCTION: ACCEPT function applying the cuts in sequence
    // -------------------------------------------------------------------------------------------------------------
    // This function checks the compatibility of a triplet with the above CA cuts by applying them in sequence.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool accept(
        [[maybe_unused]] const TAcc& acc,
        CACell<TrackerTraits> const& innerCell,
        CACell<TrackerTraits> const& outerCell,
        float& curvature,
        HitsMultiView hh,
        reco::CATripletCutsSoAConstView tripletCuts,
        reco::CATripletCutsSoAConstView::const_element tripletVectorCutsCol,
        // Row of the INNER cell's layer pair (L1,L2). Used by the beam-spot (DCA) cut on every
        // topology: its threshold is anchored at the triplet's innermost layer -- see the cut itself.
        reco::CATripletCutsSoAConstView::const_element tripletInnerPairCutsCol,
        [[maybe_unused]] reco::CAGraphSoAConstView cc,
        [[maybe_unused]] bool useTripletDNN,
        [[maybe_unused]] float tripletDNNThreshold,
#ifdef CA_TRIPLET_DUMP
        // out: the 18 BASE DNN features, filled (in DNN-block formulas, so training==deployment)
        // when the triplet is accepted; written into the TripletDump SoA at t_ind by Kernel_connect.
        float* __restrict__ dumpFeat,
        // out: the IN-KERNEL DNN score score(feat) for this triplet (in-kernel-vs-offline
        // consistency check); computed regardless of the gate in dump builds, -1 if not computed.
        float* __restrict__ dumpScore,
#endif
        uint32_t* __restrict__ pipelineCounters) {
#ifdef CA_PIPELINE_COUNTERS
      // set up the pipeline counter
      auto countRej = [&](int cut) {
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          if (pipelineCounters)
            alpaka::atomicAdd(
                acc, &pipelineCounters[caHitNtupletGenerator::kTrpRejBase + cut], 1u, alpaka::hierarchy::Blocks{});
        }
      };
#endif

      float r1 = innerCell.inner_r(hh);
      float z1 = innerCell.inner_z(hh);
      float r2 = outerCell.inner_r(hh);
      float z2 = outerCell.inner_z(hh);
      float r3 = outerCell.outer_r(hh);
      float z3 = outerCell.outer_z(hh);

      // apply alignment in RZ plane cut
      if (!alignedRZ(r1, z1, r2, z2, r3, z3, tripletCuts.ptmin(), tripletVectorCutsCol.maxRZTolerance())) {
#ifdef CA_PIPELINE_COUNTERS
        countRej(caHitNtupletGenerator::kCutAlignedRZ);
#endif
        return false;
      }

      // calculate curvature for the XY plane cuts
      float x1 = innerCell.inner_x(hh);
      float y1 = innerCell.inner_y(hh);
      float x2 = outerCell.inner_x(hh);
      float y2 = outerCell.inner_y(hh);
      float x3 = outerCell.outer_x(hh);
      float y3 = outerCell.outer_y(hh);
      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);
      curvature = eq.curvature();
      float absCurvature = std::abs(curvature);

      // apply alignment in XY plane cut
      if (!alignedXY(absCurvature, tripletCuts.maxCurv())) {
#ifdef CA_PIPELINE_COUNTERS
        countRej(caHitNtupletGenerator::kCutAlignedXY);
#endif
        return false;
      }

      // apply beamspot compatibility cut. The threshold is anchored at the triplet's innermost layer
      // L1, i.e. read from the INNER cell's layer-pair row (L1,L2), on every topology (see the
      // parameter note above); floorDCA comes from the same row.
      float tipTimesCurvature = std::abs(eq.dca0());
      if (!beamspotCompatibleXY(
              absCurvature, tipTimesCurvature, tripletInnerPairCutsCol.maxDCA(), tripletInnerPairCutsCol.floorDCA())) {
#ifdef CA_PIPELINE_COUNTERS
        countRej(caHitNtupletGenerator::kCutBeamspotCompatibleXY);
#endif
        return false;
      }

      // stub specific cuts for Phase2 OT
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        // calculate dPhi and dr values for the stubs cuts
        auto iphi1 = innerCell.inner_iphi(hh);
        auto iphi2 = outerCell.inner_iphi(hh);
        auto iphi3 = outerCell.outer_iphi(hh);
        float dPhi12 = short2phi(iphi2 - iphi1);
        float dPhi13 = short2phi(iphi3 - iphi1);
        float dPhi23 = short2phi(iphi3 - iphi2);
        float dr12 = r2 - r1;
        float dr13 = r3 - r1;

        // apply phi compatibility cut
        if (!phiCompatible(dPhi12, dr12, dPhi13, dr13, tripletCuts.maxPhiResid())) {
#ifdef CA_PIPELINE_COUNTERS
          countRej(caHitNtupletGenerator::kCutPhiCompatible);
#endif
          return false;
        }

        // calculate number of stubs
        int nStubs =
            int(innerCell.inner_isStub(hh)) + int(outerCell.inner_isStub(hh)) + int(outerCell.outer_isStub(hh));

        // Stub-curvature quantities, needed by BOTH the stub-curvature cuts AND the dump/DNN feature
        // vector below, so they are hoisted to this scope. For nStubs>0 they hold the weighted-mean
        // stub curvature and its variance. For pixel-only triplets (nStubs==0) the weighted mean would
        // be 0/0=NaN, so SENTINELS are assigned instead:
        //   curvatureStubs            = 0.f
        //   curvatureStubsErrSquared  = 1e6f  (large variance)
        // With these the derived stub pulls collapse to ~0, logErrSq becomes a fixed distinctive
        // constant and the nStubs==0 feature alone flags the regime to the DNN. The EXACT sentinel
        // values are mirrored in test/models/train_triplet_dnn.py add_derived(), so a pixel-only
        // training row is bit-consistent with what CATripletDNN.h evaluates in the kernel.
        float curvatureStubs;
        float curvatureStubsErrSquared;

        // The stub-specific cuts (same-sign dPhi + the two stub-curvature compatibility cuts) require at
        // least one stub. Pixel-only triplets SKIP those cuts and instead fall through with the two
        // sentinel values assigned below, so that they still reach the DNN gate and the dump block with a
        // well-defined feature vector instead of returning early.
        if (nStubs == 0) {
          curvatureStubs = 0.f;
          curvatureStubsErrSquared = 1e6f;  // large-variance sentinel; see the note above
        } else {
          // apply same sign dPhi cut
          if (tripletCuts.sameDPhiSign() && !sameSignDPhi(dPhi12, dPhi23)) {
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutSameSignDPhi);
#endif
            return false;
          }

          // Weighted mean stub curvature, and in the same walk the residuals against the tangent of
          // the three-point circle. The mean and its variance are DNN features and keep their trained
          // definition (the plain dPhiDrError column); the cut below uses the precision-only one.
          float sum_weights = 0.f, sum_weightsTimesCurv = 0.f;
          float residSum = 0.f, weightSum = 0.f;

          auto computeKappa = [&](uint32_t hitId, float r) {
            // Only a stub carries a bend: isStub() is false for a pixel hit and for a degenerate
            // outer-tracker row with a negative bend error.
            if (!isStub(hh, int32_t(hitId)))
              return;
            auto const stub = hh.stub(int32_t(hitId));
            float s = stub.dPhiDrError();
            float d = stub.dPhiDr();
            float den = 1.f + r * r * d * d;
            float w = den * den * den / (s * s);
            sum_weights += w;
            sum_weightsTimesCurv += w * d / std::sqrt(den);

            float sPrec = stub.dPhiDrErrorPrec();
            if (sPrec > 0.f && r > 0.f) {
              float x = hh[hitId].xGlobal(), y = hh[hitId].yGlobal();
              // Curvature error of this stub: hit precision, plus the multiple scattering that
              // separates its direction from the triplet's circle (an angle theta becomes a
              // curvature theta/r at radius r).
              float sk = sPrec / (den * std::sqrt(den));
              float sMS = caStubMS::kThetaPerCurv * absCurvature / r;
              float wPrec = 1.f / (sk * sk + sMS * sMS);
              residSum += wPrec * (d / std::sqrt(den) - circleKappaAt(eq, x, y));
              weightSum += wPrec;
            }
          };

          computeKappa(innerCell.inner_hit_id(), r1);
          computeKappa(outerCell.inner_hit_id(), r2);
          computeKappa(outerCell.outer_hit_id(), r3);

          curvatureStubs = sum_weightsTimesCurv / sum_weights;
          curvatureStubsErrSquared = 1.f / sum_weights;

          // apply compatibility with triplet cut
          if (!stubsCurvCompatibleWithTriplet(residSum, weightSum, tripletVectorCutsCol.maxStubGeomCurvSigma())) {
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutStubsCurvCompatibleWithTriplet);
#endif
            return false;
          }

          // apply compatibility with inner doublet cut
          if (!stubsCompatibleWithInnerDoublet(
                  dPhi12, dr12, curvatureStubs, tripletVectorCutsCol.maxStubInnerDoubletDCurv())) {
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutStubsCompatibleWithInnerDoublet);
#endif
            return false;
          }
        }

#ifdef CA_TRIPLET_DUMP
        // Per-BUILT-triplet dataset row (truth-labeled DNN training input), filled once per triplet
        // that passed ALL TripletCuts, from inside the Phase2OTStubs branch where the full stub
        // feature vector is in scope. The 18 BASE DNN features go into the out-param in the EXACT
        // DNN-block formulas, so the training set matches what CATripletDNN evaluates at deployment;
        // their order is BASE_FEATURES in test/models/train_triplet_dnn.py and the DERIVED features
        // are recomputed offline from these + lay1/2/3. Kernel_connect writes the row (+ the three
        // merged-hit indices h1/h2/h3, the truth join key, + layers + iter) into the TripletDump SoA.
        {
          const float dcaDump = (absCurvature > 0.f) ? tipTimesCurvature / absCurvature : 0.f;
          const float dPhiDr13Dump = dPhi13 / dr13;
          const float rmidDump = 0.5f * (r1 + r3);
          const float conv2Dump = 1.f + rmidDump * rmidDump * dPhiDr13Dump * dPhiDr13Dump;
          const float curvature13Dump = dPhiDr13Dump / std::sqrt(conv2Dump);
          dumpFeat[0] = absCurvature;
          dumpFeat[1] = tipTimesCurvature;
          dumpFeat[2] = dcaDump;
          dumpFeat[3] = curvatureStubs;
          dumpFeat[4] = curvatureStubsErrSquared;
          dumpFeat[5] = curvature13Dump;
          dumpFeat[6] = dPhi12;
          dumpFeat[7] = dPhi13;
          dumpFeat[8] = dPhi23;
          dumpFeat[9] = dr12;
          dumpFeat[10] = dr13;
          dumpFeat[11] = r1;
          dumpFeat[12] = r2;
          dumpFeat[13] = r3;
          dumpFeat[14] = z1;
          dumpFeat[15] = z2;
          dumpFeat[16] = z3;
          dumpFeat[17] = float(nStubs);
        }
#endif

        // ----------------------------------------------------------------------------
        // Inline per-triplet DNN gate (optional; compile-time weights in CATripletDNNWeights.h,
        // evaluated by CATripletDNN.h). Rejects accepted triplets whose DNN score < threshold; with
        // useTripletDNN off the block is a no-op and the cut ladder alone decides.
        // In a CA_TRIPLET_DUMP build the reject below is compiled out, so accept() returns true for every
        // cut-accepted triplet whatever useTripletDNN and whichever model is compiled in. That keeps the
        // training set unbiased: gating the dump on the compiled-in model would only ever show the next
        // DNN that model's own accepted subset. The score is still captured (dumpScore) so the in-kernel
        // evaluation can be cross-checked against the offline one.
        // Feature vector = 18 raw quantities + 11 derived (pulls/residuals/log
        // compressions/layer gaps): order AND formulas (incl. the 1e-12 eps
        // conventions) MUST match add_derived() + BASE_FEATURES/DERIVED in
        // RecoTracker/PixelSeeding/test/train_triplet_dnn_v2.py.
        // Gate regime: when the in-kernel DNN is on it gates EVERY triplet, pixel-only (nStubs==0, via
        // the sentinel features above) and stub-containing alike. One model covers both
        // regimes (nStubs is a feature); a retrained bank inherits this contract because the dump path
        // that produced its training rows is the same one.
        bool runDnnBlock = useTripletDNN;
#ifdef CA_TRIPLET_DUMP
        runDnnBlock = true;  // dump build: always evaluate the score to capture it (consistency check)
#endif
        if (runDnnBlock) {
          static_assert(caTripletDNN::kNFeat == 29, "feature vector size must match the trained MLP");
          constexpr float kEps = 1e-12f;
          const float dcaDnn = (absCurvature > 0.f) ? tipTimesCurvature / absCurvature : 0.f;
          const float dPhiDr13Dnn = dPhi13 / dr13;
          const float rmidDnn = 0.5f * (r1 + r3);
          const float conv2Dnn = 1.f + rmidDnn * rmidDnn * dPhiDr13Dnn * dPhiDr13Dnn;
          const float curvature13Dnn = dPhiDr13Dnn / std::sqrt(conv2Dnn);
          // derived features (each a handful of FLOPs vs the ~6k-MAC MLP evaluation)
          const float stubCirclePull =
              (curvatureStubs - curvature) / std::sqrt(std::max(curvatureStubsErrSquared, kEps));
          const float stubCircleRatio = curvatureStubs / (std::abs(curvature) + kEps);
          const float curv13Resid = curvature13Dnn - curvature;
          const float rzResid = z2 - (z1 + (r2 - r1) * (z3 - z1) / (dr13 + kEps));
          const float cotTheta = (z3 - z1) / (dr13 + kEps);
          const float dPhiRatio = dPhi12 / ((dPhi23 >= 0.f) ? (dPhi23 + kEps) : (dPhi23 - kEps));
          const float logAbsCurv = std::log1p(absCurvature * 1e3f);
          const float logErrSq = std::log1p(curvatureStubsErrSquared * 1e6f);
          const float logDca = std::log1p(std::abs(dcaDnn));
          const float layGap12 = float(int(outerCell.innerLayer(cc)) - int(innerCell.innerLayer(cc)));
          const float layGap23 = float(int(outerCell.outerLayer(cc)) - int(outerCell.innerLayer(cc)));
          const float feat[caTripletDNN::kNFeat] = {absCurvature,
                                                    tipTimesCurvature,
                                                    dcaDnn,
                                                    curvatureStubs,
                                                    curvatureStubsErrSquared,
                                                    curvature13Dnn,
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
                                                    static_cast<float>(nStubs),
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
          // NaN discipline, triplet half -- the same rule the track-level gate in
          // Kernel_classifyTracks follows: a non-finite quantity must never DECIDE anything. The
          // finiteness of the network INPUTS is established here, BEFORE the score is used, rather
          // than trusting a NaN to survive the MLP and the sigmoid and then to lose a comparison.
          // edm::isNotFinite is a bit-pattern test on the exponent field, so it stays valid under
          // -Ofast / -ffinite-math-only. 29 exponent tests against a ~6k-MAC evaluation: free.
          bool featFinite = true;
          for (int k = 0; featFinite && k < int(caTripletDNN::kNFeat); ++k)
            featFinite = !edm::isNotFinite(feat[k]);
          const float dnnScore = caTripletDNN_eval::score(feat);
#ifdef CA_TRIPLET_DUMP
          if (dumpScore)
            *dumpScore = dnnScore;  // capture the in-kernel score for the offline-vs-in-kernel check
          // dump build: reject compiled out so every cut-accepted triplet is dumped (see the note above).
#else
          const float defThr = caTripletDNN::kDefaultThreshold;
          const float thr = (tripletDNNThreshold >= 0.f) ? tripletDNNThreshold : defThr;
          // PROMOTING form on purpose: `score >= threshold` on finite inputs is the decision to
          // ACCEPT the triplet, and the reject is its negation -- never `if (score < thr) return
          // false`. Under -Ofast (-ffinite-math-only) the compiler may assume no NaN operand and
          // rewrite a rejecting predicate into its finite-arithmetic complement, which would let a
          // NaN score walk through the gate; in this form the default is "do not accept", so
          // anything the comparison cannot decide stays rejected. The CA_TRIPLET_DUMP early return
          // above is untouched, so the training dump keeps seeing every cut-accepted triplet.
          const bool dnnAccept = featFinite && (dnnScore >= thr);
          if (useTripletDNN && !dnnAccept)
            return false;
#endif
        }
      }

      // if we arrive at the end, the triplet passed all cuts
      return true;
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

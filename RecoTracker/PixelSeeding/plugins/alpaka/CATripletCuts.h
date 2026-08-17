#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletCuts_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletCuts_h

#include <alpaka/alpaka.hpp>
#include <cmath>

#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "CACell.h"
#include "CAPipelineCounters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  struct TripletCuts {
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

    // ---------------------------------
    // curvature compatibility with triplet aka geomKappaSigmaCut
    // ---------------------------------
    // This cut checks the compatibility of the inner-outer hit doublet's dPhi/dr with the stubs' average of the triplet.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool stubsCurvCompatibleWithTriplet(const float dPhi13,
                                                                              const float r1,
                                                                              const float r3,
                                                                              const float curvatureStubs,
                                                                              const float curvatureStubsErrSquared,
                                                                              const float maxStubGeomCurvSigma) {
      if (maxStubGeomCurvSigma < 0.f)
        return true;  // cut disabled

      float dr13 = r3 - r1;
      float rmid = 0.5f * (r1 + r3);
      float dPhiDr13 = dPhi13 / dr13;
      float conversionSquared = 1.f + rmid * rmid * dPhiDr13 * dPhiDr13;
      float curvature13 = dPhiDr13 / std::sqrt(conversionSquared);

      // Geometric curvature error (~500 murad phi resolution)
      constexpr float phiErrSquared = 25e-8f;
      float curvature13ErrSquared =
          phiErrSquared / (dr13 * dr13 * conversionSquared * conversionSquared * conversionSquared);
      float curvatureDiffErrSquared = curvatureStubsErrSquared + curvature13ErrSquared;
      float curvatureDiff = curvatureStubs - curvature13;
      bool compatible =
          curvatureDiff * curvatureDiff < maxStubGeomCurvSigma * maxStubGeomCurvSigma * curvatureDiffErrSquared;

#ifdef CA_DEBUG
      printf(
          "TripletCuts::stubsCurvCompatibleWithTriplet;dPhi13=%.4f;r1=%.4f;r3=%.4f;curvatureStubs=%.4f;"
          "curvatureStubsErrSquared=%.4e;maxStubGeomCurvSigma=%.4f;curvature13=%.4f;curvatureDiff=%.4f;"
          "curvatureDiffErrSquared=%.4e;compatible=%d\n",
          dPhi13,                    // dPhi between inner and outer hit
          r1,                        // r of inner hit
          r3,                        // r of outer hit
          curvatureStubs,            // curvature from stubs
          curvatureStubsErrSquared,  // error squared of curvature from stubs
          maxStubGeomCurvSigma,      // maximum allowed geometry curvature sigma
          curvature13,               // curvature from inner-outer hit pair
          curvatureDiff,             // computed curvature difference
          curvatureDiffErrSquared,   // error squared of curvature difference
          compatible ? 1 : 0);       // pass/fail
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

          // calculate the average stub curvature for the stubs compatibility cuts
          float sum_weights = 0.f, sum_weightsTimesCurv = 0.f;

          // Compute curvature + error for each stub hit
          auto computeKappa = [&](uint32_t hitId, float r) {
            float s = hh[hitId].dPhiDrError();
            if (s < 0.f)
              return;
            float d = hh[hitId].dPhiDr();
            float den = 1.f + r * r * d * d;
            float w = den * den * den / (s * s);
            sum_weights += w;
            sum_weightsTimesCurv += w * d / std::sqrt(den);
          };

          computeKappa(innerCell.inner_hit_id(), r1);
          computeKappa(outerCell.inner_hit_id(), r2);
          computeKappa(outerCell.outer_hit_id(), r3);

          curvatureStubs = sum_weightsTimesCurv / sum_weights;
          curvatureStubsErrSquared = 1.f / sum_weights;

          // apply compatibility with triplet cut
          if (!stubsCurvCompatibleWithTriplet(dPhi13,
                                              r1,
                                              r3,
                                              curvatureStubs,
                                              curvatureStubsErrSquared,
                                              tripletVectorCutsCol.maxStubGeomCurvSigma())) {
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
      }

      // if we arrive at the end, the triplet passed all cuts
      return true;
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

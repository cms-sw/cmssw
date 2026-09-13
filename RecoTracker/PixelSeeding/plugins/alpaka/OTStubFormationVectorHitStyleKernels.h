#ifndef RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernels_h

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace otStubFormationVectorHitStyle {
    // Parallax correction along the lower sensor's local-x axis (VectorHit convention). The
    // origin->hit vector is projected on the stack frame (zLoc = lower->upper stack direction,
    // xLoc = lower local-x made orthogonal to zLoc) and the local slope lVx / lVz is multiplied by
    // the sensor separation. Returns 0 for degenerate inputs.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeParallaxCorrection(
        TAcc const& acc,
        float gx_lower,
        float gy_lower,
        float gz_lower,
        // unit vector (global) from the lower to the upper sensor centre
        float globalLowUpNormX,
        float globalLowUpNormY,
        float globalLowUpNormZ,
        // direction of the lower sensor's local-x axis in global coordinates (unit)
        float localXInGlobalX,
        float localXInGlobalY,
        float localXInGlobalZ,
        // sensor separation, in the same length unit as the coordinates
        float separation) {
      constexpr float eps = 1e-6f;

      // gV: vector from the origin to the lower hit (global). The hit coordinates are already
      // beam-spot corrected by the converter, so the origin is the beam spot.
      float gVx = gx_lower;
      float gVy = gy_lower;
      float gVz = gz_lower;

      // zLoc direction: along the stack from LOWER -> UPPER (global, assumed normalized already)
      float zLocX = globalLowUpNormX;
      float zLocY = globalLowUpNormY;
      float zLocZ = globalLowUpNormZ;

      // Safety: ensure zLoc is valid (normalized-ish and non-zero)
      float zLocN2 = zLocX * zLocX + zLocY * zLocY + zLocZ * zLocZ;
      if (zLocN2 < eps)
        return 0.0f;

      if (alpaka::math::abs(acc, zLocN2 - 1.0f) > 1e-3f) {
        float invN = 1.0f / alpaka::math::sqrt(acc, zLocN2);
        zLocX *= invN;
        zLocY *= invN;
        zLocZ *= invN;
      }

      // xAxis (global) is the direction of LOWER local-x, but it may not be exactly orthogonal to zLoc.
      // Project it into the plane orthogonal to zLoc to get a proper in-plane x direction.
      float x0X = localXInGlobalX;
      float x0Y = localXInGlobalY;
      float x0Z = localXInGlobalZ;

      float x0N2 = x0X * x0X + x0Y * x0Y + x0Z * x0Z;
      if (x0N2 < eps)
        return 0.0f;

      if (alpaka::math::abs(acc, x0N2 - 1.0f) > 1e-3f) {
        float invN = 1.0f / alpaka::math::sqrt(acc, x0N2);
        x0X *= invN;
        x0Y *= invN;
        x0Z *= invN;
      }

      // xPerp = x0 - (x0*zLoc) zLoc
      float x0DotZ = x0X * zLocX + x0Y * zLocY + x0Z * zLocZ;
      float xPerpX = x0X - x0DotZ * zLocX;
      float xPerpY = x0Y - x0DotZ * zLocY;
      float xPerpZ = x0Z - x0DotZ * zLocZ;

      float xPerpN2 = xPerpX * xPerpX + xPerpY * xPerpY + xPerpZ * xPerpZ;
      if (xPerpN2 < eps) {
        // Degenerate: provided localX is (nearly) parallel to stack direction
        return 0.0f;
      }

      float invXPerpN = 1.0f / alpaka::math::sqrt(acc, xPerpN2);
      float xLocX = xPerpX * invXPerpN;
      float xLocY = xPerpY * invXPerpN;
      float xLocZ = xPerpZ * invXPerpN;

      // Local components needed for VectorHit-style correction:
      //   lVx = gV * xLoc
      //   lVz = gV * zLoc
      float lVx = gVx * xLocX + gVy * xLocY + gVz * xLocZ;
      float lVz = gVx * zLocX + gVy * zLocY + gVz * zLocZ;

      // Protect against near-parallel-to-plane direction (would blow up slope)
      if (alpaka::math::abs(acc, lVz) < eps) {
        return 0.0f;
      }

      float slope_x = lVx / lVz;
      float parallCorr = slope_x * separation;

      if (!alpaka::math::isfinite(acc, parallCorr)) {
        return 0.0f;
      }

      return parallCorr;  // along LOWER-sensor local_x, transporting LOWER->UPPER along globalLowUpNorm
    }

    // Acceptance of a (lower, upper) cluster pair, in one place: the counting and the formation
    // kernel must agree exactly or the count kernel reserves the wrong number of slots.
    // The width is the offline VectorHit builder's: |pC| is taken off whichever cluster is the outer
    // one by the sign of the upper local x. The signed form lx_upper - lx_lower - pC is what the
    // straight-line projection asks for, but on ttbar it removes 16-20 % of the barrel PS stubs and
    // 4.5 points of prompt barrel efficiency, so the sign convention of the parallax correction is
    // not the one that derivation assumes.
    template <typename TAcc, typename THits>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool acceptStubPair(TAcc const& acc,
                                                       THits const& hits,
                                                       uint32_t iLower,
                                                       uint32_t iUpper,
                                                       float topoLowUpNormX,
                                                       float topoLowUpNormY,
                                                       float topoLowUpNormZ,
                                                       float localXInGlobalX,
                                                       float localXInGlobalY,
                                                       float localXInGlobalZ,
                                                       float separation_cm,
                                                       float cut,
                                                       int32_t maxCSDiff,
                                                       int32_t maxCS,
                                                       int32_t maxCSSum) {
      auto const& lowerHit = hits[iLower];
      auto const& upperHit = hits[iUpper];

      if (lowerHit.yLocal() * upperHit.yLocal() < 0.f)
        return false;
      const int32_t csLower = int32_t(lowerHit.clusterSize());
      const int32_t csUpper = int32_t(upperHit.clusterSize());
      if (alpaka::math::abs(acc, csLower - csUpper) > maxCSDiff)
        return false;
      if (csLower > maxCS || csUpper > maxCS)
        return false;
      if (csLower + csUpper > maxCSSum)
        return false;

      const float pC = computeParallaxCorrection(acc,
                                                 lowerHit.xGlobal(),
                                                 lowerHit.yGlobal(),
                                                 lowerHit.zGlobal(),
                                                 topoLowUpNormX,
                                                 topoLowUpNormY,
                                                 topoLowUpNormZ,
                                                 localXInGlobalX,
                                                 localXInGlobalY,
                                                 localXInGlobalZ,
                                                 separation_cm);
      const float lx_lower = lowerHit.xLocal();
      const float lx_upper = upperHit.xLocal();
      const float absPC = alpaka::math::abs(acc, pC);
      float lpos_lower_corr = lx_lower;
      float lpos_upper_corr = lx_upper;
      if (lx_upper > lx_lower) {
        if (lx_upper > 0.f)
          lpos_upper_corr = lx_upper - absPC;
        else
          lpos_lower_corr = lx_lower + absPC;
      } else if (lx_upper < lx_lower) {
        if (lx_upper > 0.f)
          lpos_lower_corr = lx_lower - absPC;
        else
          lpos_upper_corr = lx_upper + absPC;
      } else {
        lpos_upper_corr = (lx_upper > 0.f) ? lx_upper - absPC : lx_upper + absPC;
      }
      const float width = lpos_lower_corr - lpos_upper_corr;
      return alpaka::math::abs(acc, width) < cut;
    }

    // Work division: Acc2D with Y indexing modules and X indexing warp lanes
    // (warpSize threads per module). All lanes of a warp cooperate on the
    // (iLower, iUpper) pair loop of a single module.
    class CountStubsVectorHitStyleKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
                                    int32_t const* barrelFlatMaxCSDiff,
                                    int32_t const* barrelTiltedMaxCSDiff,
                                    int32_t const* endcapMaxCSDiff,
                                    int32_t const* barrelFlatMaxCS,
                                    int32_t const* barrelTiltedMaxCS,
                                    int32_t const* endcapMaxCS,
                                    int32_t const* barrelFlatMaxCSSum,
                                    int32_t const* barrelTiltedMaxCSSum,
                                    int32_t const* endcapMaxCSSum,
                                    uint32_t* stubCounts,
                                    uint32_t nModules) const {
        // Mark as maybe_unused to suppress warning when diagnostic counters are enabled (single thread path)
        [[maybe_unused]] const int32_t warpSize = alpaka::warp::getSize(acc);
        const int32_t laneId = static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u]);

        // One warp per module
        for (uint32_t iModule : cms::alpakatools::uniform_elements_y(acc, nModules)) {
          // Get hit ranges (uniform across the warp)
          uint32_t hitStart = moduleView[iModule].moduleStart();
          uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
          uint32_t upperStart = moduleView[iModule].upperSensorStart();

          uint32_t nLower = upperStart - hitStart;
          uint32_t nUpper = hitEnd - upperStart;

          // Handle modules with no upper hits (P-hit recovery for PS modules)
          if (nUpper == 0) {
            // For PS modules: recover P-hits that have no matching S-hits.
            // These become PHitOnly stub entries (no bend measurement).
            uint8_t moduleType = geometry[iModule].moduleType();
            bool isPS = (moduleType == 0 || moduleType == 1);
            if (laneId == 0) {
              if (isPS && nLower > 0) {
                stubCounts[iModule + 1] = nLower;
              } else {
                // SS modules or no lower hits: no stubs
                stubCounts[iModule + 1] = 0;
              }
            }
            continue;
          }

          if (nLower == 0) {
            if (laneId == 0)
              stubCounts[iModule + 1] = 0;
            continue;
          }

          // Get geometry parameters (uniform across the warp)
          float separation = geometry[iModule].sensorSeparation();  // in mm
          bool isBarrel = geometry[iModule].isBarrel();
          bool isFlat = geometry[iModule].isFlat();
          [[maybe_unused]] bool isFlipped = geometry[iModule].isFlipped();
          uint8_t layer = geometry[iModule].layer();
          float globalLowUpNormX = geometry[iModule].globalLowUpNormX();
          float globalLowUpNormY = geometry[iModule].globalLowUpNormY();
          float globalLowUpNormZ = geometry[iModule].globalLowUpNormZ();
          float localXInGlobalX = geometry[iModule].localXInGlobalX();
          float localXInGlobalY = geometry[iModule].localXInGlobalY();
          float localXInGlobalZ = geometry[iModule].localXInGlobalZ();

          // Get layer-dependent cut (separate cuts for flat and tilted barrel modules)
          float cut =
              isBarrel ? (isFlat ? maxWidthBarrelFlat[layer] : maxWidthBarrelTilted[layer]) : maxWidthEndcap[layer];

          // Per-layer cluster size cuts
          int32_t maxCSDiff =
              isBarrel ? (isFlat ? barrelFlatMaxCSDiff[layer] : barrelTiltedMaxCSDiff[layer]) : endcapMaxCSDiff[layer];
          int32_t maxCS = isBarrel ? (isFlat ? barrelFlatMaxCS[layer] : barrelTiltedMaxCS[layer]) : endcapMaxCS[layer];
          int32_t maxCSSum =
              isBarrel ? (isFlat ? barrelFlatMaxCSSum[layer] : barrelTiltedMaxCSSum[layer]) : endcapMaxCSSum[layer];

          // Topological lower->upper normal (flipped sign for flipped modules; see note in
          // formation kernel below)
          float flipSign = isFlipped ? -1.0f : 1.0f;
          float topoLowUpNormX = flipSign * globalLowUpNormX;
          float topoLowUpNormY = flipSign * globalLowUpNormY;
          float topoLowUpNormZ = flipSign * globalLowUpNormZ;
          float separation_cm = separation * 0.1f;

          // Warp-cooperative count: lanes stride across the flattened pair index
          // p = (iLower - hitStart) * nUpper + (iUpper - upperStart).
          // Output ordering is irrelevant here -- we only need the total per module.
          uint32_t nPairs = nLower * nUpper;
          int32_t localCount = 0;
          for (uint32_t p = static_cast<uint32_t>(laneId); p < nPairs; p += warpSize) {
            uint32_t iLower = hitStart + p / nUpper;
            uint32_t iUpper = upperStart + p % nUpper;
            if (acceptStubPair(acc,
                               hits,
                               iLower,
                               iUpper,
                               topoLowUpNormX,
                               topoLowUpNormY,
                               topoLowUpNormZ,
                               localXInGlobalX,
                               localXInGlobalY,
                               localXInGlobalZ,
                               separation_cm,
                               cut,
                               maxCSDiff,
                               maxCS,
                               maxCSSum))
              ++localCount;
          }

          // Warp-reduce localCount across lanes (xor-butterfly).
          // All lanes must be active for the shuffles to work, so no early return above.
          for (int32_t off = 1; off < warpSize; off <<= 1) {
            localCount += alpaka::warp::shfl_xor(acc, localCount, off);
          }
          if (laneId == 0) {
            stubCounts[iModule + 1] = static_cast<uint32_t>(localCount);
          }
        }
      }
    };

    // Formation kernel for VectorHits style.
    // Work division: Acc2D with Y indexing modules and X indexing warp lanes
    // (warpSize threads per module). All lanes of a warp cooperate on the
    // (iLower, iUpper) pair loop of a single module. Lanes consume pairs in their
    // natural lexicographic order; a warp exclusive prefix scan assigns each
    // passing lane an output slot, so the per-module stubs[] write order is
    // the lexicographic pair order, independent of how the lanes are scheduled.
    class FormStubsVectorHitStyleKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
                                    int32_t const* barrelFlatMaxCSDiff,
                                    int32_t const* barrelTiltedMaxCSDiff,
                                    int32_t const* endcapMaxCSDiff,
                                    int32_t const* barrelFlatMaxCS,
                                    int32_t const* barrelTiltedMaxCS,
                                    int32_t const* endcapMaxCS,
                                    int32_t const* barrelFlatMaxCSSum,
                                    int32_t const* barrelTiltedMaxCSSum,
                                    int32_t const* endcapMaxCSSum,
                                    uint32_t const* stubOffsets,
                                    ::reco::StubsView stubs,
                                    uint32_t nModules) const {
        const int32_t warpSize = alpaka::warp::getSize(acc);
        const int32_t laneId = static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u]);

        // One warp per module
        for (uint32_t iModule : cms::alpakatools::uniform_elements_y(acc, nModules)) {
          // Get hit ranges (uniform across the warp)
          uint32_t hitStart = moduleView[iModule].moduleStart();
          uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
          uint32_t upperStart = moduleView[iModule].upperSensorStart();

          uint32_t nLower = upperStart - hitStart;
          uint32_t nUpper = hitEnd - upperStart;

          // Get geometry parameters (uniform across the warp)
          float separation = geometry[iModule].sensorSeparation();  // in mm
          uint8_t moduleType = geometry[iModule].moduleType();
          bool isPS = (moduleType == 0 || moduleType == 1);
          bool isBarrel = geometry[iModule].isBarrel();
          bool isFlat = geometry[iModule].isFlat();
          bool isFlipped = geometry[iModule].isFlipped();
          uint8_t layer = geometry[iModule].layer();
          float globalLowUpNormX = geometry[iModule].globalLowUpNormX();
          float globalLowUpNormY = geometry[iModule].globalLowUpNormY();
          float globalLowUpNormZ = geometry[iModule].globalLowUpNormZ();
          float localXInGlobalX = geometry[iModule].localXInGlobalX();
          float localXInGlobalY = geometry[iModule].localXInGlobalY();
          float localXInGlobalZ = geometry[iModule].localXInGlobalZ();

          float cut =
              isBarrel ? (isFlat ? maxWidthBarrelFlat[layer] : maxWidthBarrelTilted[layer]) : maxWidthEndcap[layer];

          // Per-layer cluster size cuts
          int32_t maxCSDiff =
              isBarrel ? (isFlat ? barrelFlatMaxCSDiff[layer] : barrelTiltedMaxCSDiff[layer]) : endcapMaxCSDiff[layer];
          int32_t maxCS = isBarrel ? (isFlat ? barrelFlatMaxCS[layer] : barrelTiltedMaxCS[layer]) : endcapMaxCS[layer];
          int32_t maxCSSum =
              isBarrel ? (isFlat ? barrelFlatMaxCSSum[layer] : barrelTiltedMaxCSSum[layer]) : endcapMaxCSSum[layer];

          uint32_t stubBase = stubOffsets[iModule];
          uint32_t maxStubIdx = stubOffsets[iModule + 1];  // End offset for this module

          // Handle PHitOnly recovery for PS modules with P-hits but no S-hits
          if (nUpper == 0) {
            if (isPS && nLower > 0) {
              // Create PHitOnly stub entries for each P-hit (lanes stride across P-hits)
              // These have position but no bend measurement.
              for (uint32_t k = static_cast<uint32_t>(laneId); k < nLower; k += warpSize) {
                uint32_t out = stubBase + k;
                if (out >= maxStubIdx)
                  continue;
                uint32_t iLower = hitStart + k;
                auto const& pHit = hits[iLower];

                float xg = pHit.xGlobal();
                float yg = pHit.yGlobal();
                stubs[out].iphi() = unsafe_atan2s<7>(yg, xg);

                // Stub-specific fields set to invalid values (no bend measurement).
                // Negative dPhiDrError tags PHitOnly entries (see reco::isStub).
                stubs[out].dPhiDr() = 0.0f;
                stubs[out].dPhiDrError() = -1.0f;
                stubs[out].dPhiDrErrorPrec() = -1.0f;  // same sentinel: no bend measurement

                stubs[out].posHitIdx() = iLower;
                stubs[out].lowerHitIdx() = iLower;
                stubs[out].upperHitIdx() = UINT32_MAX;  // Invalid marker

                stubs[out].flags() = ::reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer, isPS);
              }
            }
            continue;
          }

          if (nLower == 0)
            continue;

          // Topological lower->upper normal: globalLowUpNorm is the physical inner->outer
          // direction, while the width calculation needs the topological lower->upper direction,
          // so the sign is flipped for flipped modules.
          float flipSign = isFlipped ? -1.0f : 1.0f;
          float topoLowUpNormX = flipSign * globalLowUpNormX;
          float topoLowUpNormY = flipSign * globalLowUpNormY;
          float topoLowUpNormZ = flipSign * globalLowUpNormZ;
          float separation_cm = separation * 0.1f;

          // Tilt parameters and sensor frames (uniform across the warp)
          float sinTilt = geometry[iModule].sinTilt();
          float cosTilt = geometry[iModule].cosTilt();
          auto const innerFrame =
              isFlipped ? geometry[iModule].upperSensorFrame() : geometry[iModule].lowerSensorFrame();
          auto const outerFrame =
              isFlipped ? geometry[iModule].lowerSensorFrame() : geometry[iModule].upperSensorFrame();

          // Warp-cooperative formation. Lanes stride across the flattened pair index
          // p = (iLower - hitStart) * nUpper + (iUpper - upperStart) in their natural
          // (iLower, iUpper) order. A warp prefix scan over the per-lane
          // result assigns each passing lane an output slot, preserving the original
          // ordering of stubs[] writes.
          uint32_t nPairs = nLower * nUpper;
          uint32_t writeIdx = stubBase;  // running output cursor, uniform across the warp

          for (uint32_t pBase = 0; pBase < nPairs; pBase += warpSize) {
            uint32_t p = pBase + static_cast<uint32_t>(laneId);
            bool predicate = false;
            uint32_t iLower = 0;
            uint32_t iUpper = 0;

            if (p < nPairs) {
              iLower = hitStart + p / nUpper;
              iUpper = upperStart + p % nUpper;
              predicate = acceptStubPair(acc,
                                         hits,
                                         iLower,
                                         iUpper,
                                         topoLowUpNormX,
                                         topoLowUpNormY,
                                         topoLowUpNormZ,
                                         localXInGlobalX,
                                         localXInGlobalY,
                                         localXInGlobalZ,
                                         separation_cm,
                                         cut,
                                         maxCSDiff,
                                         maxCS,
                                         maxCSSum);
            }

            // Warp exclusive prefix scan of (predicate ? 1 : 0).
            // All lanes must execute the shuffles regardless of their predicate or the
            // bound check, so the warp stays converged through this section.
            int32_t myCount = predicate ? 1 : 0;
            int32_t inclusive = myCount;
            for (int32_t off = 1; off < warpSize; off <<= 1) {
              int32_t y = alpaka::warp::shfl(acc, inclusive, laneId - off);
              if (laneId >= off)
                inclusive += y;
            }
            int32_t myOffset = inclusive - myCount;
            int32_t totalThisIter = alpaka::warp::shfl(acc, inclusive, warpSize - 1);

            if (predicate) {
              uint32_t out = writeIdx + static_cast<uint32_t>(myOffset);
              if (out < maxStubIdx) {
                // Determine physical inner/outer for stub properties (position, hit
                // indices, errors).
                uint32_t innerIdx = isFlipped ? iUpper : iLower;
                uint32_t outerIdx = isFlipped ? iLower : iUpper;

                // Contract: a stub stores exactly one sensor's position+error, and the same
                // sensor's azimuth. PS: the pixel sensor (better spatial precision), topologically
                // lower on a PSP (moduleType == 0) and upper on a PSS. 2S: the physically inner
                // sensor, which on a flipped stack is not the topologically lower one.
                uint32_t pickedIdx = isPS ? ((moduleType == 0) ? iLower : iUpper) : innerIdx;
                float xg = hits[pickedIdx].xGlobal();
                float yg = hits[pickedIdx].yGlobal();
                float zg = hits[pickedIdx].zGlobal();
                float rg = alpaka::math::sqrt(acc, xg * xg + yg * yg);

                // iPhi - same encoding as pixel hits (16-bit signed, full circle = [-32768, 32767])
                stubs[out].iphi() = unsafe_atan2s<7>(yg, xg);

                float xi = hits[innerIdx].xGlobal();
                float yi = hits[innerIdx].yGlobal();
                float xo = hits[outerIdx].xGlobal();
                float yo = hits[outerIdx].yGlobal();

                // Zero-bend reference of the pair: a straight track from the beam spot has the same azimuth on both
                // sensors, so the other hit is compared with the point where the ray through the position hit crosses the
                // other sensor's plane, and it is moved along its own local y (the centre of its strip segment) to the
                // crossing point's along-strip coordinate. Wherever local y has a radial component (tilted barrel, discs)
                // an along-strip offset dy enters the azimuth difference as x * dy / r^2, larger than the bend itself for
                // a strip centre. The pair's along-strip error becomes the position hit's.
                const uint32_t otherIdx = (pickedIdx == innerIdx) ? outerIdx : innerIdx;
                auto const& rotOther = (otherIdx == innerIdx) ? innerFrame.rotation() : outerFrame.rotation();
                float yerrInnerEff = hits[innerIdx].yerrLocal();
                float yerrOuterEff = hits[outerIdx].yerrLocal();
                float rayScale = 1.0f;
                {
                  const float xOther = hits[otherIdx].xGlobal();
                  const float yOther = hits[otherIdx].yGlobal();
                  const float zOther = hits[otherIdx].zGlobal();
                  const float nDotPick = globalLowUpNormX * xg + globalLowUpNormY * yg + globalLowUpNormZ * zg;
                  const float nDotOther =
                      globalLowUpNormX * xOther + globalLowUpNormY * yOther + globalLowUpNormZ * zOther;
                  // the ray crosses the other sensor's plane at t times the position hit
                  const float t = (alpaka::math::abs(acc, nDotPick) > 1e-6f) ? nDotOther / nDotPick : 1.0f;
                  rayScale = t;
                  const float dy = rotOther.yx() * (t * xg - xOther) + rotOther.yy() * (t * yg - yOther) +
                                   rotOther.yz() * (t * zg - zOther);
                  if (otherIdx == innerIdx) {
                    xi += dy * rotOther.yx();
                    yi += dy * rotOther.yy();
                    yerrInnerEff = hits[pickedIdx].yerrLocal();
                  } else {
                    xo += dy * rotOther.yx();
                    yo += dy * rotOther.yy();
                    yerrOuterEff = hits[pickedIdx].yerrLocal();
                  }
                }

                float ri2 = xi * xi + yi * yi;
                float ro2 = xo * xo + yo * yo;

                // The bend is the azimuth difference itself. On the ray a straight track leaves none, on
                // any surface, so no parallax term is subtracted here: the local-x parallax belongs to
                // the width test, which compares local x rather than azimuth.
                float phi_inner = alpaka::math::atan2(acc, yi, xi);
                float phi_outer = alpaka::math::atan2(acc, yo, xo);
                float ri = alpaka::math::sqrt(acc, ri2);
                float ro = alpaka::math::sqrt(acc, ro2);

                float dphi = phi_outer - phi_inner;
                if (dphi > M_PI)
                  dphi -= 2.0f * M_PI;
                if (dphi < -M_PI)
                  dphi += 2.0f * M_PI;

                float dr_geometric = ro - ri;
                float denominator = cosTilt + sinTilt * zg / rg;

                float dr_effective;
                if (alpaka::math::abs(acc, denominator) > 1e-6f) {
                  dr_effective = separation_cm / denominator;
                } else {
                  dr_effective = dr_geometric;
                }

                // A degenerate geometry leaves no bend measurement: tag the entry as a non-stub
                // rather than publishing a zero bend with perfect precision.
                const bool bendValid = alpaka::math::abs(acc, dr_effective) > 1e-6f;
                if (!bendValid) {
                  stubs[out].dPhiDr() = 0.0f;
                  stubs[out].dPhiDrError() = -1.0f;
                  stubs[out].dPhiDrErrorPrec() = -1.0f;
                } else {
                  const float bend = dphi / dr_effective;
                  stubs[out].dPhiDr() = bend;

                  // dPhiDr error: propagate per-sensor local errors to global via the
                  // sensor frames, then form the global phi variance.
                  float ge_i[6], ge_o[6];
                  innerFrame.toGlobal(hits[innerIdx].xerrLocal(), 0.0f, yerrInnerEff, ge_i);
                  outerFrame.toGlobal(hits[outerIdx].xerrLocal(), 0.0f, yerrOuterEff, ge_o);

                  float sinphi_i = alpaka::math::sin(acc, phi_inner);
                  float cosphi_i = alpaka::math::cos(acc, phi_inner);
                  float sig2phi_i = (ri2 > 0.0f)
                                        ? (sinphi_i * sinphi_i * ge_i[0] - 2.0f * sinphi_i * cosphi_i * ge_i[1] +
                                           cosphi_i * cosphi_i * ge_i[2]) /
                                              ri2
                                        : 0.0f;
                  float sinphi_o = alpaka::math::sin(acc, phi_outer);
                  float cosphi_o = alpaka::math::cos(acc, phi_outer);
                  float sig2phi_o = (ro2 > 0.0f)
                                        ? (sinphi_o * sinphi_o * ge_o[0] - 2.0f * sinphi_o * cosphi_o * ge_o[1] +
                                           cosphi_o * cosphi_o * ge_o[2]) /
                                              ro2
                                        : 0.0f;
                  float dphi_err = alpaka::math::sqrt(acc, sig2phi_i + sig2phi_o);
                  stubs[out].dPhiDrError() = dphi_err / alpaka::math::abs(acc, dr_effective);

                  // Precision-only azimuthal bend error: the three coordinates the pair measures (two local x readings, one
                  // along-strip coordinate: the macro-pixel on PS, the strip centre on 2S), propagated into
                  // dPhiDr = dphi / dr_effective. Where the strip has a radial component (discs, tilted barrel) the
                  // along-strip coordinate moves both azimuths and the radial lever dr_effective = separation /
                  // (cosTilt + sinTilt z/r); on a disc both are -dPhiDr/r, so the bend carries a relative error of
                  // 2 sigma_r / r (3 to 5 % for a 2S disc stub).
                  {
                    auto const& rotI = innerFrame.rotation();
                    auto const& rotO = outerFrame.rotation();
                    auto const& rotPick = (pickedIdx == innerIdx) ? rotI : rotO;
                    const float riInv = (ri > 1e-6f) ? 1.0f / ri : 0.0f;
                    const float roInv = (ro > 1e-6f) ? 1.0f / ro : 0.0f;
                    // azimuthal angular sensitivity to a unit displacement along each local axis
                    const float uxI = (cosphi_i * rotI.xy() - sinphi_i * rotI.xx()) * riInv;
                    const float uxO = (cosphi_o * rotO.xy() - sinphi_o * rotO.xx()) * roInv;
                    const float uyI = (cosphi_i * rotI.yy() - sinphi_i * rotI.yx()) * riInv;
                    float uyO = (cosphi_o * rotO.yy() - sinphi_o * rotO.yx()) * roInv;
                    // orient the outer strip direction onto the inner one: the two local frames' y-axes
                    // may be flipped relative to each other without any physical meaning, and the two
                    // azimuths have to move against one common sense of the along-strip coordinate
                    const float dotY = rotI.yx() * rotO.yx() + rotI.yy() * rotO.yy() + rotI.yz() * rotO.yz();
                    if (dotY < 0.0f)
                      uyO = -uyO;
                    // the other hit rides the ray, so it travels rayScale times as far along its strip
                    const float dPhiDy = (pickedIdx == innerIdx) ? (rayScale * uyO - uyI) : (uyO - rayScale * uyI);
                    // the same coordinate moves the divisor: d(ln dr_effective)/dy, with the strip's
                    // radial and longitudinal components at the position hit, taken in the same sense
                    // as the azimuth term above (which is oriented on the inner sensor's y axis)
                    const float pickSense = (pickedIdx == innerIdx) ? 1.0f : ((dotY < 0.0f) ? -1.0f : 1.0f);
                    const float yRad = pickSense * (rotPick.yx() * xg + rotPick.yy() * yg) / rg;
                    const float yLong = pickSense * rotPick.yz();
                    const float dLever = (alpaka::math::abs(acc, denominator) > 1e-6f)
                                             ? sinTilt * (yLong * rg - zg * yRad) / (denominator * rg * rg)
                                             : 0.0f;
                    const float dBendDy = dPhiDy / dr_effective + bend * dLever;
                    const float vxI = alpaka::math::max(acc, hits[innerIdx].xerrLocal(), 0.0f);
                    const float vxO = alpaka::math::max(acc, hits[outerIdx].xerrLocal(), 0.0f);
                    const float vyPick = alpaka::math::max(acc, hits[pickedIdx].yerrLocal(), 0.0f);
                    const float sig2prec = (vxI * uxI * uxI + vxO * uxO * uxO) / (dr_effective * dr_effective) +
                                           dBendDy * dBendDy * vyPick;
                    const float errPrec = alpaka::math::sqrt(acc, alpaka::math::max(acc, sig2prec, 0.0f));
                    stubs[out].dPhiDrErrorPrec() = errPrec;
                  }
                }

                stubs[out].posHitIdx() = pickedIdx;
                stubs[out].lowerHitIdx() = iLower;
                stubs[out].upperHitIdx() = iUpper;
                stubs[out].flags() = ::reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer, isPS);
              }
            }

            writeIdx += static_cast<uint32_t>(totalThisIter);
          }
        }
      }
    };

  }  // namespace otStubFormationVectorHitStyle

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernels_h

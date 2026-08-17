#ifndef RecoTracker_PixelSeeding_plugins_alpaka_Phase2OTVectorHitsFormationKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_Phase2OTVectorHitsFormationKernels_h

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace phase2OTVectorHitsFormation {
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

    // Acceptance of a (lower, upper) cluster pair
    template <typename TAcc, typename THits>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool acceptVectorHitPair(TAcc const& acc,
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
                                                            float cut) {
      auto const& lowerHit = hits[iLower];
      auto const& upperHit = hits[iUpper];

      if (lowerHit.yLocal() * upperHit.yLocal() < 0.f)
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
    class CountVectorHitsKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
                                    uint32_t* vectorHitCounts,
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
            // These become PHitOnly vector hit entries (no bend measurement).
            uint8_t moduleType = geometry[iModule].moduleType();
            bool isPS = (moduleType == 0 || moduleType == 1);
            if (laneId == 0) {
              if (isPS && nLower > 0) {
                vectorHitCounts[iModule + 1] = nLower;
              } else {
                // SS modules or no lower hits: no vector hits
                vectorHitCounts[iModule + 1] = 0;
              }
            }
            continue;
          }

          if (nLower == 0) {
            if (laneId == 0)
              vectorHitCounts[iModule + 1] = 0;
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
            if (acceptVectorHitPair(acc,
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
                                    cut))
              ++localCount;
          }

          // Warp-reduce localCount across lanes (xor-butterfly).
          // All lanes must be active for the shuffles to work, so no early return above.
          for (int32_t off = 1; off < warpSize; off <<= 1) {
            localCount += alpaka::warp::shfl_xor(acc, localCount, off);
          }
          if (laneId == 0) {
            vectorHitCounts[iModule + 1] = static_cast<uint32_t>(localCount);
          }
        }
      }
    };

    // Formation kernel for VectorHits style.
    // Work division: Acc2D with Y indexing modules and X indexing warp lanes
    // (warpSize threads per module). All lanes of a warp cooperate on the
    // (iLower, iUpper) pair loop of a single module. Lanes consume pairs in their
    // natural lexicographic order; a warp exclusive prefix scan assigns each
    // passing lane an output slot, so the per-module vectorHits[] write order is
    // the lexicographic pair order, independent of how the lanes are scheduled.
    class FormVectorHitsKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
                                    uint32_t const* vectorHitOffsets,
                                    ::reco::VectorHitsView vectorHits,
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

          uint32_t vectorHitBase = vectorHitOffsets[iModule];
          uint32_t maxVectorHitIdx = vectorHitOffsets[iModule + 1];  // End offset for this module

          // Handle PHitOnly recovery for PS modules with P-hits but no S-hits
          if (nUpper == 0) {
            if (isPS && nLower > 0) {
              // Create PHitOnly vector hit entries for each P-hit (lanes stride across P-hits)
              // These have position but no bend measurement.
              for (uint32_t k = static_cast<uint32_t>(laneId); k < nLower; k += warpSize) {
                uint32_t out = vectorHitBase + k;
                if (out >= maxVectorHitIdx)
                  continue;
                uint32_t iLower = hitStart + k;
                auto const& pHit = hits[iLower];

                float xg = pHit.xGlobal();
                float yg = pHit.yGlobal();
                vectorHits[out].iphi() = unsafe_atan2s<7>(yg, xg);

                // Position, errors and CA module of the published hit.
                vectorHits[out].xGlobal() = xg;
                vectorHits[out].yGlobal() = yg;
                vectorHits[out].zGlobal() = pHit.zGlobal();
                vectorHits[out].rGlobal() = alpaka::math::sqrt(acc, xg * xg + yg * yg);
                vectorHits[out].xerrLocal() = pHit.xerrLocal();
                vectorHits[out].yerrLocal() = pHit.yerrLocal();
                vectorHits[out].detectorIndex() = pHit.detectorIndex();

                // Vector-hit-specific fields set to invalid values (no bend measurement).
                // Negative dPhiDrError tags PHitOnly entries (see reco::isVectorHit).
                vectorHits[out].dPhiDr() = 0.0f;
                vectorHits[out].dPhiDrError() = -1.0f;

                vectorHits[out].posHitIdx() = iLower;
                vectorHits[out].lowerHitIdx() = iLower;
                vectorHits[out].upperHitIdx() = UINT32_MAX;  // Invalid marker

                vectorHits[out].flags() = ::reco::VectorHitFlags::makeFlags(isBarrel, isFlat, true, layer, isPS);
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
          // ordering of vectorHits[] writes.
          uint32_t nPairs = nLower * nUpper;
          uint32_t writeIdx = vectorHitBase;  // running output cursor, uniform across the warp

          for (uint32_t pBase = 0; pBase < nPairs; pBase += warpSize) {
            uint32_t p = pBase + static_cast<uint32_t>(laneId);
            bool predicate = false;
            uint32_t iLower = 0;
            uint32_t iUpper = 0;

            if (p < nPairs) {
              iLower = hitStart + p / nUpper;
              iUpper = upperStart + p % nUpper;
              predicate = acceptVectorHitPair(acc,
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
                                              cut);
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
              if (out < maxVectorHitIdx) {
                // Determine physical inner/outer for vector hit properties (position, hit
                // indices, errors).
                uint32_t innerIdx = isFlipped ? iUpper : iLower;
                uint32_t outerIdx = isFlipped ? iLower : iUpper;

                // Contract: a vector hit stores exactly one sensor's position+error, and the same
                // sensor's azimuth. PS: the pixel sensor (better spatial precision), topologically
                // lower on a PSP (moduleType == 0) and upper on a PSS. 2S: the physically inner
                // sensor, which on a flipped stack is not the topologically lower one.
                uint32_t pickedIdx = isPS ? ((moduleType == 0) ? iLower : iUpper) : innerIdx;
                float xg = hits[pickedIdx].xGlobal();
                float yg = hits[pickedIdx].yGlobal();
                float zg = hits[pickedIdx].zGlobal();
                float rg = alpaka::math::sqrt(acc, xg * xg + yg * yg);

                // iPhi - same encoding as pixel hits (16-bit signed, full circle = [-32768, 32767])
                vectorHits[out].iphi() = unsafe_atan2s<7>(yg, xg);

                // Position, errors and CA module of the published hit; rGlobal reuses rg.
                vectorHits[out].xGlobal() = xg;
                vectorHits[out].yGlobal() = yg;
                vectorHits[out].zGlobal() = zg;
                vectorHits[out].rGlobal() = rg;
                vectorHits[out].xerrLocal() = hits[pickedIdx].xerrLocal();
                vectorHits[out].yerrLocal() = hits[pickedIdx].yerrLocal();
                vectorHits[out].detectorIndex() = hits[pickedIdx].detectorIndex();

                // Zero-bend reference of the pair: the ray from the beam spot through the position hit P, which crosses
                // the other sensor's plane at t X_P. A straight track from the beam spot crosses it there.
                const uint32_t otherIdx = (pickedIdx == innerIdx) ? outerIdx : innerIdx;
                auto const& rotOther = (otherIdx == innerIdx) ? innerFrame.rotation() : outerFrame.rotation();
                float rayScale = 1.0f;
                // Strip-count bend: the other hit's local x minus the local x of P's projection t X_P on the other
                // sensor. rotOther's first row is the other sensor's local-x axis e_x in global coordinates, and a
                // point's local x is e_x . (X - C) with C the sensor centre; for the difference of two points on
                // that sensor C cancels:  bend = x_other - x_proj = e_x . (X_other - t X_P).
                // Only global hit positions (beam-spot corrected) enter, never C. The straight-line parallax is
                // inside the projection, and nothing along the strips enters.
                float bendLocal = 0.0f;
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
                  bendLocal = rotOther.xx() * (xOther - t * xg) + rotOther.xy() * (yOther - t * yg) +
                              rotOther.xz() * (zOther - t * zg);
                }

                const float xi = hits[innerIdx].xGlobal(), yi = hits[innerIdx].yGlobal();
                const float xo = hits[outerIdx].xGlobal(), yo = hits[outerIdx].yGlobal();
                float dr_geometric =
                    alpaka::math::sqrt(acc, xo * xo + yo * yo) - alpaka::math::sqrt(acc, xi * xi + yi * yi);
                float denominator = cosTilt + sinTilt * zg / rg;

                float dr_effective;
                if (alpaka::math::abs(acc, denominator) > 1e-6f) {
                  dr_effective = separation_cm / denominator;
                } else {
                  dr_effective = dr_geometric;
                }

                // A degenerate geometry leaves no bend measurement: tag the entry as a non-vector-hit
                // rather than publishing a zero bend with perfect precision.
                const bool bendValid = alpaka::math::abs(acc, dr_effective) > 1e-6f;
                if (!bendValid) {
                  vectorHits[out].dPhiDr() = 0.0f;
                  vectorHits[out].dPhiDrError() = -1.0f;
                } else {
                  // dPhiDr = sign * u_x * bend / dr_effective. u_x = cos(alpha) / r_proj turns a local-x distance on
                  // the other sensor into an azimuth difference, r_proj = t * rg being the radius of P's projection
                  // there. The sign makes dPhiDr = (phi_outer - phi_inner) / dr_effective.
                  const float rgInv = (rg > 1e-6f) ? 1.0f / rg : 0.0f;
                  const float rHatX = xg * rgInv, rHatY = yg * rgInv;  // r_hat at P = at the projection
                  const float phiHatX = -rHatY, phiHatY = rHatX;       // phi_hat at P = at the projection
                  // alpha: the angle in the transverse plane between the other sensor's local-x axis e_x and
                  // phi_hat at P (0 for a hit on the module centre line). cos(alpha) < 0 when e_x points along
                  // -phi_hat; the sign carries into u_x and dPhiDr.
                  const float cosAlpha = phiHatX * rotOther.xx() + phiHatY * rotOther.xy();  // phi_hat . e_x(other)
                  const float sinAlpha = rHatX * rotOther.xx() + rHatY * rotOther.xy();      // r_hat . e_x(other)
                  const float rProj = rayScale * rg;
                  const float uxProj = (rProj > 1e-6f) ? cosAlpha / rProj : 0.0f;
                  const float qSign = (otherIdx == outerIdx) ? 1.0f : -1.0f;
                  const float bend = qSign * uxProj * bendLocal / dr_effective;
                  vectorHits[out].dPhiDr() = bend;

                  // Bend error, by propagation of the three measured coordinates into
                  // dPhiDr = K * bend, K = sign * u_x / dr_effective.
                  {
                    auto const& rotPick = (pickedIdx == innerIdx) ? innerFrame.rotation() : outerFrame.rotation();
                    const float dotX =
                        rotOther.xx() * rotPick.xx() + rotOther.xy() * rotPick.xy() + rotOther.xz() * rotPick.xz();
                    const float vxPick = alpaka::math::max(acc, hits[pickedIdx].xerrLocal(), 0.0f);
                    const float vxOther = alpaka::math::max(acc, hits[otherIdx].xerrLocal(), 0.0f);
                    const float vyPick = alpaka::math::max(acc, hits[pickedIdx].yerrLocal(), 0.0f);
                    const float K = qSign * uxProj / dr_effective;
                    const float sig2count = K * K * (vxOther + rayScale * rayScale * dotX * dotX * vxPick);
                    const float phiDotEy = phiHatX * rotPick.yx() + phiHatY * rotPick.yy();
                    const float rDotEy = rHatX * rotPick.yx() + rHatY * rotPick.yy();
                    const float tanAlpha = (alpaka::math::abs(acc, cosAlpha) > 1e-6f) ? sinAlpha / cosAlpha : 0.0f;
                    const float dLnUx = -(tanAlpha * phiDotEy + rDotEy) * rgInv;
                    const float dLnDr = (alpaka::math::abs(acc, denominator) > 1e-6f)
                                            ? -sinTilt * (rotPick.yz() * rg - zg * rDotEy) / (denominator * rg * rg)
                                            : 0.0f;
                    const float dLnK = dLnUx - dLnDr;
                    const float sig2pos = bend * bend * dLnK * dLnK * vyPick;
                    const float errPrec = alpaka::math::sqrt(acc, alpaka::math::max(acc, sig2count + sig2pos, 0.0f));
                    vectorHits[out].dPhiDrError() = errPrec;
                  }
                }

                vectorHits[out].posHitIdx() = pickedIdx;
                vectorHits[out].lowerHitIdx() = iLower;
                vectorHits[out].upperHitIdx() = iUpper;
                vectorHits[out].flags() = ::reco::VectorHitFlags::makeFlags(isBarrel, isFlat, true, layer, isPS);
              }
            }

            writeIdx += static_cast<uint32_t>(totalThisIter);
          }
        }
      }
    };

  }  // namespace phase2OTVectorHitsFormation

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_Phase2OTVectorHitsFormationKernels_h

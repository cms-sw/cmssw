#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "RecoTracker/PixelSeeding/interface/CAVectorHitMS.h"
#include "CACell.h"
#include "CAStructures.h"

// #define GPU_DEBUG
// #define DOUBLETS_DEBUG  // Very verbose - enable only for detailed debugging
// #define CA_WARNINGS

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {
  using namespace cms::alpakatools;
  using namespace ::caStructures;
  using namespace ::reco;

  using HitToCell = GenericContainer;

  template <typename TrackerTraits>
  using PhiBinner = PhiBinnerT<TrackerTraits>;
  //Move this ^ definition in CAStructures maybe

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadderPhase1(int const moduleId) {
    return (0 == (moduleId / 8) % 2);
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadderPhase2(int const moduleId) {
    return (0 != (moduleId / 18) % 2);
  }

  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder(const int moduleId);

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase1>(int const moduleId) {
    return moduleIsOuterLadderPhase1(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::HIonPhase1>(int const moduleId) {
    return moduleIsOuterLadderPhase1(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase2>(int const moduleId) {
    return moduleIsOuterLadderPhase2(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase2OT>(int const moduleId) {
    return moduleIsOuterLadderPhase2(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase2OTVectorHits>(int const moduleId) {
    return moduleIsOuterLadderPhase2(moduleId);
  }

  template <typename TrackerTraits, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool dSizeCut(const TAcc& acc,
                                               caStructures::HitsViewT<TrackerTraits> hh,
                                               ::reco::CALayersSoAConstView ll,
                                               ::reco::CADoubletCutsSoAConstView doubletCuts,
                                               int i,
                                               int o) {
    const uint32_t mi = hh[i].detectorIndex();
    const uint32_t mo = hh[o].detectorIndex();
    const auto first_forward = ll.layerStarts()[4];
    const auto first_bpix2 = ll.layerStarts()[1];
    bool innerB1 = mi < first_bpix2;
    bool isOuterLadder = moduleIsOuterLadder<TrackerTraits>(mi);
    auto innerBarrel = mi < first_forward;
    auto onlyBarrel = mo < first_forward;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d mi = %d innerB1 = %d isOuterLadder = %d first_forward = %d first_bpix2 = %d\n",
           i,
           o,
           mi,
           innerB1,
           isOuterLadder,
           first_forward,
           first_bpix2);
    printf("i = %d o = %d mo = %d innerB1 = %d isOuterLadder = %d \n", i, o, mo, innerBarrel, onlyBarrel);
#endif
    // The module/barrel predicates are decided before any cluster size is read: a cluster size is a
    // pixel-only column, which an outer-tracker entry does not have.
    // An inner hit on an inner-barrel-1 inner ladder and a pair with no barrel hit are rejected here;
    // an outer-tracker inner hit sits beyond every barrel pixel module and its outer partner further
    // out still, so it never reaches the cluster-size reads.
    if (not innerBarrel and not onlyBarrel) {
      return false;
    }
    if (innerB1 and not isOuterLadder) {
      return false;
    }

    // A cluster size is stored signed and is negative when the cluster touches a sensor edge, so
    // `mes < 0` also rejects edge clusters.
    auto mes = int(caStructures::clusterSizeY(hh, i));
    if (mes < 0) {
      return false;
    }

    auto dz = hh[i].zGlobal() - hh[o].zGlobal();
    auto dr = hh[i].rGlobal() - hh[o].rGlobal();
    auto dy = innerB1 ? doubletCuts.maxDSizeB1() : doubletCuts.maxDSize();
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d dy = %d maxDSizeB1 = %d maxDSize = %d dzdrFact = %.2f maxDSizePred = %d \n",
           i,
           o,
           dy,
           doubletCuts.maxDSizeB1(),
           doubletCuts.maxDSize(),
           doubletCuts.dzdrFact(),
           doubletCuts.maxDSizePred());
#endif
    if (onlyBarrel) {
      // The outer cluster size is read only here, where the outer hit is a barrel pixel one.
      auto so = caStructures::clusterSizeY(hh, o);
      return so > 0 && std::abs(so - mes) > dy;
    }
    return innerBarrel &&
           std::abs(mes - int(std::abs(dz / dr) * doubletCuts.dzdrFact() + 0.5f)) > doubletCuts.maxDSizePred();
  }

  template <typename TrackerTraits, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool clusterCut(const TAcc& acc,
                                                 caStructures::HitsViewT<TrackerTraits> hh,
                                                 ::reco::CALayersSoAConstView ll,
                                                 ::reco::CADoubletCutsSoAConstView doubletCuts,
                                                 uint32_t i) {
    const uint32_t mi = hh[i].detectorIndex();
    const auto first_bpix2 = ll.layerStarts()[1];
    const auto first_bpix3 = ll.layerStarts()[2];
    bool innerB1orB2 = mi < ll.layerStarts()[2];
#ifdef DOUBLETS_DEBUG
    printf(
        "i = %d mi = %d innerB1orB2 = %d innerB1 = %d innerB2 = %d minYsizeB1 = %d minYsizeB2 = %d isOuterLadder = %d "
        "mes = %d \n",
        i,
        mi,
        innerB1orB2,
        mi < first_bpix2,
        (mi >= first_bpix2) && (mi < first_bpix3),
        doubletCuts.minInnerSizeB1(),
        doubletCuts.minInnerSizeB2(),
        (0 == (mi / 8) % 2),
        innerB1orB2 && ((!(mi < first_bpix2)) || (0 == (mi / 8) % 2)) ? int(caStructures::clusterSizeY(hh, i)) : -1);
#endif
    if (!innerB1orB2)
      return false;

    bool innerB1 = mi < first_bpix2;

    bool isOuterLadder = moduleIsOuterLadder<TrackerTraits>(mi);
    // innerB1orB2 above established that the hit is on a barrel pixel module, so the pixel-only
    // cluster-size column exists.
    auto mes = (!innerB1) || isOuterLadder ? int(caStructures::clusterSizeY(hh, i)) : -1;

    if (innerB1)  // B1
      if (mes > 0 && mes < doubletCuts.minInnerSizeB1()) {
        return true;  // only long cluster  (5*8)
      }
    bool innerB2 = (mi >= first_bpix2) && (mi < first_bpix3);
    if (innerB2)  // B2 and F1
      if (mes > 0 && mes < doubletCuts.minInnerSizeB2()) {
        return true;
      }

    return false;
  }

  // CountOnly: run the full cut path but only increment nCells (do not write the cell
  // or the hit->cell association). Used by the
  // countDoubletsFirst runtime flag's first pass to learn the exact nCells before allocating
  template <typename TrackerTraits, bool CountOnly, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void doubletsFromHisto(const TAcc& acc,
                                                        uint32_t maxNumOfDoublets,
                                                        CACell<TrackerTraits>* cells,
                                                        uint32_t* nCells,
                                                        caStructures::HitsViewT<TrackerTraits> hh,
                                                        ::reco::CAGraphSoAConstView cc,
                                                        ::reco::CALayersSoAConstView ll,
                                                        ::reco::CADoubletCutsSoAConstView doubletCuts,
                                                        uint32_t const* __restrict__ offsets,
                                                        PhiBinner<TrackerTraits> const* phiBinner,
                                                        HitToCell* outerHitHisto) {
    const bool doClusterCut = doubletCuts.minInnerSizeB1() > 0 or doubletCuts.minInnerSizeB2() > 0;
    const bool doZSizeCut =
        doubletCuts.maxDSizeB1() > 0 or doubletCuts.maxDSize() > 0 or doubletCuts.maxDSizePred() > 0;

    const uint32_t nPairs = cc.metadata().size();
    using PhiHisto = PhiBinner<TrackerTraits>;
    ALPAKA_ASSERT_ACC(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairs for the OT-extended CA is of order 100 (bounded by TrackerTraits::nPairs; the bound is
    // enforced by CAHitNtupletAlpaka::globalBeginRun, which rejects a geometry.pairGraph exceeding it).
    // If it should become much bigger than 64, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    auto& innerLayerCumulativeSize = alpaka::declareSharedVar<uint32_t[TrackerTraits::nPairs], __COUNTER__>(acc);
    auto& ntot = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

#ifdef DOUBLETS_DEBUG
    if (cms::alpakatools::once_per_grid(acc))
      printf(
          "maxNumDoublets = %d  cc.metadata().size() = %d ll.metadata().size() = %d dzdrFact = %.2f "
          "doClusterCut = %d doZSizeCut = %d\n",
          maxNumOfDoublets,
          cc.metadata().size(),
          ll.metadata().size(),
          doubletCuts.dzdrFact(),
          doClusterCut,
          doZSizeCut);
#endif

    if (cms::alpakatools::once_per_block(acc)) {
      innerLayerCumulativeSize[0] = layerSize(cc.layerPair()[0][0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(cc.layerPair()[i][0]);
      }
      ntot = innerLayerCumulativeSize[nPairs - 1];
    }
    alpaka::syncBlockThreads(acc);

    // declared outside the loop, as it cannot go backward
    uint32_t pairLayerId = 0;

    // outermost parallel loop, using all grid elements along the slower dimension (Y or 0 in a 2D grid)
    for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, ntot)) {
      // move to lower_bound ?
      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;

      ALPAKA_ASSERT_ACC(pairLayerId < nPairs);
      ALPAKA_ASSERT_ACC(j < innerLayerCumulativeSize[pairLayerId]);
      ALPAKA_ASSERT_ACC(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = cc.layerPair()[pairLayerId][0];
      uint8_t outer = cc.layerPair()[pairLayerId][1];
      ALPAKA_ASSERT_ACC(outer > inner);

      auto hoff = PhiHisto::histOff(outer);
      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      ALPAKA_ASSERT_ACC(i >= offsets[inner]);
      ALPAKA_ASSERT_ACC(i < offsets[inner + 1]);
#ifdef DOUBLETS_DEBUG
      printf("pairLayerId = %d i = %d inner = %d outer = %d offsets[inner] = %d offsets[inner + 1] = %d\n",
             pairLayerId,
             i,
             inner,
             outer,
             offsets[inner],
             offsets[inner + 1]);
#endif
      // found hit corresponding to our worker thread, now do the job
      if (hh[i].detectorIndex() > ll.layerStarts()[ll.metadata().size() - 1]) {  //TODO use cc
#ifdef DOUBLETS_DEBUG
        printf("Killed here 1\n");
#endif
        continue;  // invalid
      }

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto zi = hh[i].zGlobal();
      auto ri = hh[i].rGlobal();

      // cut on inner coordinate (z or r depending on layer)
      auto valInner = ll.isBarrel()[inner] ? zi : ri;
      if (valInner < doubletCuts.minInner()[pairLayerId] || valInner > doubletCuts.maxInner()[pairLayerId]) {
#ifdef DOUBLETS_DEBUG
        printf("Killed here 2 --> valInner: %f [index: %d], minInner: %f, maxInner: %f\n",
               valInner,
               hh[i].detectorIndex(),
               doubletCuts.minInner()[pairLayerId],
               doubletCuts.maxInner()[pairLayerId]);
#endif
        continue;
      }

#ifdef DOUBLETS_DEBUG
      if (doClusterCut && outer > pixelTopology::last_barrel_layer)
        printf("clustCut: %d %d \n", i, clusterCut<TrackerTraits, TAcc>(acc, hh, ll, doubletCuts, i));
#endif

      if (doClusterCut && outer > pixelTopology::last_barrel_layer &&
          clusterCut<TrackerTraits, TAcc>(acc, hh, ll, doubletCuts, i)) {
#ifdef DOUBLETS_DEBUG
        printf("Killed here 4\n");
#endif
        continue;
      }

      auto mep = hh[i].iphi();

#ifdef DOUBLETS_DEBUG
      if (inner >= 28) {  // Only print for OT layers
        printf("Inner hit: idx=%d layer=%d iphi=%d detIndex=%d zi=%.2f ri=%.2f\n",
               i,
               inner,
               mep,
               hh[i].detectorIndex(),
               zi,
               ri);
      }
#endif

      // all cuts: true if fails
      auto ptcut = [&](int j, int16_t idphi) {
        // ptCut already converted to minRadius2T4 in CAHitNtupletGenerator.cc
        auto ro = hh[j].rGlobal();
        auto dphi = short2phi(idphi);
        return dphi * dphi * (doubletCuts.minPt()[pairLayerId] - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh[j].zGlobal();
        auto ro = hh[j].rGlobal();
        auto dr = ro - ri;
#ifdef DOUBLETS_DEBUG
        printf("dr: %4.3f, %4.3f, %4.3f --> %d\n", ri, ro, dr, (dr > doubletCuts.maxDR()[pairLayerId]));
        printf("zi: %4.3f, zo: %4.3f, std::abs((zi * ro - ri * zo)): %4.3f --> %d\n",
               zi,
               zo,
               std::abs((zi * ro - ri * zo)),
               (std::abs((zi * ro - ri * zo)) > doubletCuts.maxZ0()[pairLayerId] * dr));
#endif
        return dr > doubletCuts.maxDR()[pairLayerId] || dr < 0 ||
               std::abs((zi * ro - ri * zo)) > doubletCuts.maxZ0()[pairLayerId] * dr;
      };

      auto iphicut = doubletCuts.maxDPhi()[pairLayerId];

      auto kl = PhiHisto::bin(int16_t(mep - iphicut));
      auto kh = PhiHisto::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % PhiHisto::nbins(); };

#ifdef GPU_DEBUG
      // Only print for first few pairs to avoid flooding
      if (pairLayerId < 5 && i == 0) {
        auto innerLayer = cc.layerPair()[pairLayerId][0];
        auto outerLayer = cc.layerPair()[pairLayerId][1];
        printf(
            "[CAPixelDoublets] Pair %d: layers (%d->%d) | phiCut=%d | minIn=%.1f maxIn=%.1f | minOut=%.1f maxOut=%.1f "
            "| maxDR=%.1f | minDZ=%.1f maxDZ=%.1f\n",
            pairLayerId,
            innerLayer,
            outerLayer,
            doubletCuts.maxDPhi()[pairLayerId],
            doubletCuts.minInner()[pairLayerId],
            doubletCuts.maxInner()[pairLayerId],
            doubletCuts.minOuter()[pairLayerId],
            doubletCuts.maxOuter()[pairLayerId],
            doubletCuts.maxDR()[pairLayerId],
            doubletCuts.minDZ()[pairLayerId],
            doubletCuts.maxDZ()[pairLayerId]);
      }
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
        //#ifdef GPU_DEBUG
        //        if (kk != kl && kk != kh)
        //          nmin += phiBinner->size(kk + hoff);
        //#endif

        auto const* __restrict__ p = phiBinner->begin(kk + hoff);
        auto const* __restrict__ e = phiBinner->end(kk + hoff);
        auto const maxpIndex = e - p;

#ifdef DOUBLETS_DEBUG
        if (outer >= 28) {  // Only print for OT layers
          printf("PhiBinner search: inner=%d outer=%d kk=%d hoff=%d phiBin=%d maxpIndex=%d\n",
                 inner,
                 outer,
                 kk,
                 hoff,
                 kk + hoff,
                 int(maxpIndex));
        }
#endif

        // innermost parallel loop, using the block elements along the faster dimension (X or 1 in a 2D grid)
        for (uint32_t pIndex : cms::alpakatools::independent_group_elements_x(acc, maxpIndex)) {
          // FIXME implement alpaka::ldg and use it here? or is it const* __restrict__ enough?
          auto oi = p[pIndex];
          ALPAKA_ASSERT_ACC(oi >= offsets[outer]);
          ALPAKA_ASSERT_ACC(oi < offsets[outer + 1]);
#ifdef DOUBLETS_DEBUG
          printf("Exploring couple i: %d o: %d\n", i, oi);
#endif
          auto mo = hh[oi].detectorIndex();

          // invalid - use TrackerTraits::numberOfModules for correct limit per tracker configuration
          if (mo >= TrackerTraits::numberOfModules) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 4 --> mo: %d >= numberOfModules: %d\n", mo, TrackerTraits::numberOfModules);
#endif
            continue;
          }

          auto zo = hh[oi].zGlobal();
          auto ro = hh[oi].rGlobal();

          // cut on outer coordinate (z or r depending on layer)
          auto valOuter = ll.isBarrel()[outer] ? zo : ro;
          if (valOuter < doubletCuts.minOuter()[pairLayerId] || valOuter > doubletCuts.maxOuter()[pairLayerId]) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 5 --> valOuter: %f [index: %d], minOuter: %f, maxOuter: %f\n",
                   valOuter,
                   mo,
                   doubletCuts.minOuter()[pairLayerId],
                   doubletCuts.maxOuter()[pairLayerId]);
#endif
            continue;
          }

          auto dz = zo - zi;

          // cut on signed dz
          if (dz < doubletCuts.minDZ()[pairLayerId] || dz > doubletCuts.maxDZ()[pairLayerId]) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 5 --> dz: %f [index: %d], minDZ: %f, maxDZ: %f\n",
                   dz,
                   mo,
                   doubletCuts.minDZ()[pairLayerId],
                   doubletCuts.maxDZ()[pairLayerId]);
#endif
            continue;
          }

          // The z0 cut (and, with it, the dr window it carries) is applied only where the per-pair
          // z0 cut is enabled, i.e. positive. The scalar cellZ0Cut is broadcast onto this column
          // for every pair (see CAHitNtuplet.cc) unless cellZ0CutPerPair overrides it.
          if (doubletCuts.maxZ0()[pairLayerId] > 0.f && z0cutoff(oi)) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 5\n");
#endif
            continue;
          }

          auto mop = hh[oi].iphi();
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          if (idphi > iphicut) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 6 --> idphi: %d, iphicut: %d\n", idphi, iphicut);
#endif
            continue;
          }
#ifdef DOUBLETS_DEBUG
          printf("dSizeCut: %d %d %d \n", i, oi, dSizeCut<TrackerTraits, TAcc>(acc, hh, ll, doubletCuts, i, oi));
#endif
          if (doZSizeCut && dSizeCut<TrackerTraits, TAcc>(acc, hh, ll, doubletCuts, i, oi)) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 7\n");
#endif
            continue;
          }

          if (doubletCuts.minPt()[pairLayerId] > 0. && ptcut(oi, idphi)) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 8\n");
#endif
            continue;
          }

          // Vector-hit-pair cuts. Direction: the directions measured by the two vector hits must be those of a
          // circle through both vector hits, within maxVectorHitCurvSigma standard deviations (negative = off).
          if constexpr (std::is_same_v<pixelTopology::Phase2OTVectorHits, TrackerTraits>) {
            auto vectorHitSigmaCut = doubletCuts.maxVectorHitCurvSigma()[pairLayerId];
            if (vectorHitSigmaCut > 0.f && ll.isOT()[inner] && isVectorHit(hh, oi) && isVectorHit(hh, i)) {
              float d_i = hh.vectorHit(i).dPhiDr(), s_i = hh.vectorHit(i).dPhiDrError();
              float d_o = hh.vectorHit(oi).dPhiDr(), s_o = hh.vectorHit(oi).dPhiDrError();
              float xi = hh[i].xGlobal(), yi = hh[i].yGlobal();
              float xo = hh[oi].xGlobal(), yo = hh[oi].yGlobal();

              float den_i = 1.f + ri * ri * d_i * d_i;
              float den_o = 1.f + ro * ro * d_o * d_o;
              float tix = xi - ri * d_i * yi, tiy = ri * d_i * xi + yi;
              float tox = xo - ro * d_o * yo, toy = ro * d_o * xo + yo;
              float cx = xo - xi, cy = yo - yi;

              float sinI = cx * tiy - cy * tix, cosI = cx * tix + cy * tiy;
              float sinO = cx * toy - cy * tox, cosO = cx * tox + cy * toy;
              float normI = ri * std::sqrt(den_i), normO = ro * std::sqrt(den_o);
              float chord2 = cx * cx + cy * cy;
              float norm = chord2 * normI * normO;

              float resid = (norm > 0.f) ? (sinI * cosO + cosI * sinO) / norm : 0.f;
              float sbi = ri * s_i / den_i, sbo = ro * s_o / den_o;
              // Multiple scattering between the two layers.
              float kappaEst = std::abs(d_i * ri / normI + d_o * ro / normO);
              float sMS = caVectorHitMS::kThetaPerCurv * kappaEst;
              float combined_err2 = sbi * sbi + sbo * sbo + sMS * sMS;
              float significance = std::abs(resid) / std::sqrt(combined_err2);

              // Bending: the two vector hits must measure the same bend, within the errors and the difference
              // allowed by maxVectorHitTip.
              float k_i = d_i / std::sqrt(den_i), k_o = d_o / std::sqrt(den_o);
              float sk_i = s_i / (den_i * std::sqrt(den_i)), sk_o = s_o / (den_o * std::sqrt(den_o));
              float drPair = ro - ri;
              float sMSk = (drPair > 0.f) ? sMS / drPair : 0.f;  // an angle over the lever arm is a curvature
              float tipBound = doubletCuts.maxVectorHitTip() * std::abs(1.f / (ri * ri) - 1.f / (ro * ro));
              float curvWindow = vectorHitSigmaCut * std::sqrt(sk_i * sk_i + sk_o * sk_o + sMSk * sMSk) + tipBound;

              if (significance > vectorHitSigmaCut || std::abs(k_i - k_o) > curvWindow) {
#ifdef DOUBLETS_DEBUG
                printf("Killed here 10: vector hit sigma cut (sig=%.2f > cut=%.2f, barrel_i=%d barrel_o=%d)\n",
                       significance,
                       vectorHitSigmaCut,
                       (int)hh.vectorHit(i).isBarrel(),
                       (int)hh.vectorHit(oi).isBarrel());
#endif
                continue;
              }
            }
          }

          // Pixel-vector-hit cut: the bend measured by the outer vector hit must match that of the line from the
          // pixel hit to the vector hit, within maxVectorHitCurvSigma standard deviations (negative = off).
          if constexpr (std::is_same_v<pixelTopology::Phase2OTVectorHits, TrackerTraits>) {
            auto vectorHitSigmaCut = doubletCuts.maxVectorHitCurvSigma()[pairLayerId];
            if (vectorHitSigmaCut > 0.f && !ll.isOT()[inner] && isVectorHit(hh, oi)) {
              auto signed_dphi = short2phi(int16_t(mop - mep));
              auto dr = ro - ri;
              if (dr > 0.f) {
                float dphidr_doublet = signed_dphi / dr;
                float den_d = 1.f + ro * ro * dphidr_doublet * dphidr_doublet;
                float sqrt_den_d = std::sqrt(den_d);
                float k_doublet = dphidr_doublet / sqrt_den_d;

                float d_o = hh.vectorHit(oi).dPhiDr(), s_o = hh.vectorHit(oi).dPhiDrError();
                float den_o = 1.f + ro * ro * d_o * d_o;
                float sqrt_den_o = std::sqrt(den_o);
                float k_vector_hit = d_o / sqrt_den_o;
                float sk_vector_hit = s_o / (den_o * sqrt_den_o);

                float tipTerm = doubletCuts.maxVectorHitTip() * dr / (ri * ro * ro);
                // Multiple scattering between the two layers.
                float msTerm = caVectorHitMS::kThetaPerCurv * 2.f * std::abs(k_vector_hit) / dr;
                float err2 = sk_vector_hit * sk_vector_hit + tipTerm * tipTerm + msTerm * msTerm;
                if (err2 > 0.f) {
                  float significance = std::abs(k_doublet - k_vector_hit) / std::sqrt(err2);
                  if (significance > vectorHitSigmaCut) {
#ifdef DOUBLETS_DEBUG
                    printf("Killed here 11: pixel-vector-hit kappa cut (sig=%.2f > cut=%.2f)\n",
                           significance,
                           vectorHitSigmaCut);
#endif
                    continue;
                  }
                }
              }
            }
          }

          auto ind = alpaka::atomicAdd(acc, nCells, 1u, alpaka::hierarchy::Blocks{});
          // In CountOnly mode the hit->cell storage is not yet allocated, so only the
          // maxNumOfDoublets safety cap applies (the capacity() term is short-circuited).
          if (ind >= maxNumOfDoublets or (!CountOnly and ind >= uint32_t(outerHitHisto->capacity()))) {
#ifdef CA_WARNINGS
            printf("Warning!!!! Too many cells (maxNumOfDoublets = %d - nHitsToCell = %d)!\n",
                   maxNumOfDoublets,
                   outerHitHisto->capacity());
#endif
            alpaka::atomicSub(acc, nCells, 1u, alpaka::hierarchy::Blocks{});
            break;
          }

          // Count-only pass: the cell index is reserved; skip writing the cell and the
          // hit->cell association.
          if constexpr (CountOnly) {
            continue;
          }

          // Key-range guard: one bin per outer hit. Dropping here instead of writing outside off[]
          // keeps a sizing mismatch non-corrupting; the drop is counted on the matching fill pass
          // (FillDoubletsHisto), which skips exactly the same keys.
          if ((oi - hh.view(0).offsetBPIX2()) < outerHitHisto->nOnes()) {
            outerHitHisto->count(acc, oi - hh.view(0).offsetBPIX2());
          }
          cells[ind].init(hh, pairLayerId, i, oi);
#ifdef DOUBLETS_DEBUG
          printf("doublet: %d layerPair: %d inner: %d outer: %d i: %d oi: %d\n", ind, pairLayerId, inner, outer, i, oi);
#endif
        }
      }
    }  // loop in block...
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h

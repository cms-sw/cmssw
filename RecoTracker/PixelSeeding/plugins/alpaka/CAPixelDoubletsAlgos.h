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
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "CACell.h"
#include "CAStructures.h"
#include "CAHitNtupletGeneratorKernels.h"

// #define GPU_DEBUG
// #define DOUBLETS_DEBUG
// #define CA_WARNINGS

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {
  using namespace cms::alpakatools;
  using namespace ::caStructures;
  using namespace ::reco;

  using HitToCell = GenericContainer;

  template <typename TrackerTraits>
  using PhiBinner = PhiBinnerT<TrackerTraits>;
  //Move this ^ definition in CAStructures maybe

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool zSizeCut(
      const TAcc& acc, HitsConstView hh, ::reco::CALayersSoAConstView ll, AlgoParams const& params, int i, int o) {
    const uint32_t mi = hh[i].detectorIndex();
    const auto first_forward = ll.layerStarts()[4];
    const auto first_bpix2 = ll.layerStarts()[1];
    bool innerB1 = mi < first_bpix2;
    bool isOuterLadder = 0 == (mi / 8) % 2;
    auto mes = (!innerB1) || isOuterLadder ? hh[i].clusterSizeY() : -1;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d mi = %d innerB1 = %d isOuterLadder = %d first_forward = %d first_bpix2 = %d\n",
           i,
           o,
           mi,
           innerB1,
           isOuterLadder,
           first_forward,
           first_bpix2);
#endif
    if (mes < 0)
      return false;

    const uint32_t mo = hh[o].detectorIndex();
    auto so = hh[o].clusterSizeY();

    auto dz = hh[i].zGlobal() - hh[o].zGlobal();
    auto dr = hh[i].rGlobal() - hh[o].rGlobal();

    auto innerBarrel = mi < first_forward;
    auto onlyBarrel = mo < first_forward;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d mo = %d innerB1 = %d isOuterLadder = %d \n", i, o, mo, innerBarrel, onlyBarrel);
#endif
    if (not innerBarrel and not onlyBarrel)
      return false;
    auto dy = innerB1 ? params.maxDYsize12_ : params.maxDYsize_;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d dy = %d maxDYsize12_ = %d maxDYsize_ = %d dzdrFact_ = %.2f maxDYPred_ = %d \n",
           i,
           o,
           dy,
           params.maxDYsize12_,
           params.maxDYsize_,
           params.dzdrFact_,
           params.maxDYPred_);
#endif
    return onlyBarrel
               ? so > 0 && std::abs(so - mes) > dy
               : innerBarrel && std::abs(mes - int(std::abs(dz / dr) * params.dzdrFact_ + 0.5f)) > params.maxDYPred_;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool clusterCut(
      const TAcc& acc, HitsConstView hh, ::reco::CALayersSoAConstView ll, AlgoParams const& params, uint32_t i) {
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
        params.minYsizeB1_,
        params.minYsizeB2_,
        (0 == (mi / 8) % 2),
        (!(mi < first_bpix2)) || (0 == (mi / 8) % 2) ? hh[i].clusterSizeY() : -1);
#endif
    if (!innerB1orB2)
      return false;

    bool innerB1 = mi < first_bpix2;

    bool isOuterLadder = 0 == (mi / 8) % 2;
    auto mes = (!innerB1) || isOuterLadder ? hh[i].clusterSizeY() : -1;

    if (innerB1)  // B1
      if (mes > 0 && mes < params.minYsizeB1_)
        return true;  // only long cluster  (5*8)
    bool innerB2 = (mi >= first_bpix2) && (mi < first_bpix3);
    if (innerB2)  // B2 and F1
      if (mes > 0 && mes < params.minYsizeB2_)
        return true;

    return false;
  }

  template <typename TrackerTraits, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void doubletsFromHisto(const TAcc& acc,
                                                        uint32_t maxNumOfDoublets,
                                                        CACell<TrackerTraits>* cells,
                                                        uint32_t* nCells,
                                                        HitsConstView hh,
                                                        ::reco::CAGraphSoAConstView cc,
                                                        ::reco::CALayersSoAConstView ll,
                                                        uint32_t const* __restrict__ offsets,
                                                        PhiBinner<TrackerTraits> const* phiBinner,
                                                        HitToCell* outerHitHisto,
                                                        AlgoParams const& params) {
    const bool doClusterCut = params.minYsizeB1_ > 0 or params.minYsizeB2_ > 0;
    const bool doZSizeCut = params.maxDYsize12_ > 0 or params.maxDYsize_ > 0 or params.maxDYPred_ > 0;

    // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
    const float minRadius = params.cellPtCut_ * 87.78f;
    const float minRadius2T4 = 4.f * minRadius * minRadius;

    const uint32_t nPairs = cc.metadata().size();
    using PhiHisto = PhiBinner<TrackerTraits>;
    ALPAKA_ASSERT_ACC(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should much be bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    auto& innerLayerCumulativeSize = alpaka::declareSharedVar<uint32_t[pixelTopology::maxPairs], __COUNTER__>(acc);
    auto& ntot = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

#ifdef DOUBLETS_DEBUG
    if (cms::alpakatools::once_per_grid(acc))
      printf(
          "maxNumDoublets = %d  cc.metadata().size() = %d ll.metadata().size() = %d cellZ0Cut_ = %.2f cellPtCut_ = "
          "%.2f doClusterCut = %d doZ0Cut = %d  doPtCut = %d doZSizeCut = %d\n",
          maxNumOfDoublets,
          cc.metadata().size(),
          ll.metadata().size(),
          params.cellZ0Cut_,
          params.cellPtCut_,
          doClusterCut,
          params.cellZ0Cut_ > 0,
          params.cellPtCut_ > 0,
          doZSizeCut);
#endif

    if (cms::alpakatools::once_per_block(acc)) {
      innerLayerCumulativeSize[0] = layerSize(cc.graph()[0][0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(cc.graph()[i][0]);
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

      uint8_t inner = cc.graph()[pairLayerId][0];
      uint8_t outer = cc.graph()[pairLayerId][1];
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
      if (hh[i].detectorIndex() > ll.layerStarts()[ll.metadata().size() - 1])  //TODO use cc
        continue;                                                              // invalid

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto mez = hh[i].zGlobal();

      if (mez < cc.minz()[pairLayerId] || mez > cc.maxz()[pairLayerId])
        continue;
#ifdef DOUBLETS_DEBUG
      if (doClusterCut && outer > pixelTopology::last_barrel_layer)
        printf("clustCut: %d %d \n", i, clusterCut<TAcc>(acc, hh, ll, params, i));
#endif

      if (doClusterCut && outer > pixelTopology::last_barrel_layer && clusterCut<TAcc>(acc, hh, ll, params, i))
        continue;

      auto mep = hh[i].iphi();
      auto mer = hh[i].rGlobal();

      // all cuts: true if fails
      auto ptcut = [&](int j, int16_t idphi) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh[j].rGlobal();
        auto dphi = short2phi(idphi);
        return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh[j].zGlobal();
        auto ro = hh[j].rGlobal();
        auto dr = ro - mer;
        return dr > cc.maxr()[pairLayerId] || dr < 0 || std::abs((mez * ro - mer * zo)) > params.cellZ0Cut_ * dr;
      };

      auto iphicut = cc.phiCuts()[pairLayerId];

      auto kl = PhiHisto::bin(int16_t(mep - iphicut));
      auto kh = PhiHisto::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % PhiHisto::nbins(); };

#ifdef GPU_DEGBU
      printf("pairLayerId %d %d %.2f %.2f %.2f \n",
             pairLayerId,
             cc.phiCuts()[pairLayerId],
             cc.maxr()[pairLayerId],
             cc.maxz()[pairLayerId],
             cc.minz()[pairLayerId]);
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

        // innermost parallel loop, using the block elements along the faster dimension (X or 1 in a 2D grid)
        for (uint32_t pIndex : cms::alpakatools::independent_group_elements_x(acc, maxpIndex)) {
          // FIXME implement alpaka::ldg and use it here? or is it const* __restrict__ enough?
          auto oi = p[pIndex];
          ALPAKA_ASSERT_ACC(oi >= offsets[outer]);
          ALPAKA_ASSERT_ACC(oi < offsets[outer + 1]);
          auto mo = hh[oi].detectorIndex();

          // invalid
          if (mo > pixelClustering::maxNumModules)  //FIXME use cc?
            continue;

          if (params.cellZ0Cut_ > 0. && z0cutoff(oi))
            continue;

          auto mop = hh[oi].iphi();
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          if (idphi > iphicut)
            continue;
#ifdef DOUBLETS_DEBUG
          printf("zSizeCut: %d %d %d \n", i, oi, zSizeCut<TAcc>(acc, hh, ll, params, i, oi));
#endif
          if (doZSizeCut && zSizeCut<TAcc>(acc, hh, ll, params, i, oi))
            continue;

          if (params.cellPtCut_ > 0. && ptcut(oi, idphi))
            continue;

          auto ind = alpaka::atomicAdd(acc, nCells, 1u, alpaka::hierarchy::Blocks{});
          if (ind >= maxNumOfDoublets) {
#ifdef CA_WARNINGS
            printf("Warning!!!! Too many cells (limit = %d)!\n", maxNumOfDoublets);
#endif
            alpaka::atomicSub(acc, nCells, 1u, alpaka::hierarchy::Blocks{});
            break;
          }

          outerHitHisto->count(acc, oi - hh.offsetBPIX2());
          cells[ind].init(hh, pairLayerId, inner, outer, i, oi);
#ifdef DOUBLETS_DEBUG
          printf("doublet: %d layerPair: %d inner: %d outer: %d i: %d oi: %d\n", ind, pairLayerId, inner, outer, i, oi);
#endif
        }
      }
    }  // loop in block...
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h

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
#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"

#include "CACell.h"
#include "CAStructures.h"
#include "CAHitNtupletGeneratorKernels.h"

//#define GPU_DEBUG
//#define NTUPLE_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {
  using namespace cms::alpakatools;
  using namespace ::caStructures;
  using namespace ::reco;

  template <typename TrackerTraits>
  using CellNeighbors = CellNeighborsT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracks = CellTracksT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellNeighborsVector = CellNeighborsVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracksVector = CellTracksVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using OuterHitOfCell = OuterHitOfCellT<TrackerTraits>;
  
  template <typename TrackerTraits>
  using PhiBinner = cms::alpakatools::HistoContainer<int16_t,
                                                    256,
                                                    -1, 
                                                    8 * sizeof(int16_t),
                                                    typename TrackerTraits::hindex_type,
                                                    TrackerTraits::numberOfLayers>; 

  template <typename T, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool __attribute__((always_inline)) zSizeCut(const TAcc& acc,
                                                                              HitsConstView hh,
                                                                              int i,
                                                                              int o) {
    const uint32_t mi = hh[i].detectorIndex();
    const bool idealConditions_ = false;
    bool innerB1 = mi < T::last_bpix1_detIndex;
    bool isOuterLadder = idealConditions_ ? true : 0 == (mi / 8) % 2;
    auto mes = (!innerB1) || isOuterLadder ? hh[i].clusterSizeY() : -1;

    if (mes < 0)
      return false;

    const uint32_t mo = hh[o].detectorIndex();
    auto so = hh[o].clusterSizeY();

    auto dz = hh[i].zGlobal() - hh[o].zGlobal();
    auto dr = hh[i].rGlobal() - hh[o].rGlobal();

    auto innerBarrel = mi < T::last_barrel_detIndex;
    auto onlyBarrel = mo < T::last_barrel_detIndex;

    if (not innerBarrel and not onlyBarrel)
      return false;
    auto dy = innerB1 ? T::maxDYsize12 : T::maxDYsize;

    return onlyBarrel ? so > 0 && std::abs(so - mes) > dy
                      : innerBarrel && std::abs(mes - int(std::abs(dz / dr) * T::dzdrFact + 0.5f)) > T::maxDYPred;
  }

  template <typename T, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool __attribute__((always_inline)) clusterCut(const TAcc& acc,
                                                                                HitsConstView hh,
                                                                                uint32_t i) {
    const uint32_t mi = hh[i].detectorIndex();
    bool innerB1orB2 = mi < T::last_bpix2_detIndex;

    if (!innerB1orB2)
      return false;

    bool innerB1 = mi < T::last_bpix1_detIndex;
    const bool idealConditions_ = false;
    bool isOuterLadder = idealConditions_ ? true : 0 == (mi / 8) % 2;
    auto mes = (!innerB1) || isOuterLadder ? hh[i].clusterSizeY() : -1;

    if (innerB1)  // B1
      if (mes > 0 && mes < T::minYsizeB1)
        return true;                                                                 // only long cluster  (5*8)
    bool innerB2 = (mi >= T::last_bpix1_detIndex) && (mi < T::last_bpix2_detIndex);  //FIXME number
    if (innerB2)                                                                     // B2 and F1
      if (mes > 0 && mes < T::minYsizeB2)
        return true;

    return false;
  }
  

  template <typename TrackerTraits, typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void __attribute__((always_inline)) doubletsFromHisto(
      const TAcc& acc,
      CACellT<TrackerTraits>* cells,
      uint32_t* nCells,
      CellNeighborsVector<TrackerTraits>* cellNeighbors,
      CellTracksVector<TrackerTraits>* cellTracks,
      HitsConstView hh,
      ::reco::CACellsSoAConstView cc,
      uint32_t const* __restrict__ offsets,
      PhiBinner<TrackerTraits>* phiBinner,
      OuterHitOfCell<TrackerTraits> isOuterHitOfCell,
      AlgoParams const& params) {  

    // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
    const float minRadius = cc.cellPtCut() * 87.78f;
    const float minRadius2T4 = 4.f * minRadius * minRadius;

    const auto maxNumOfDoublets = params.maxNumberOfDoublets_;
  
    const uint32_t nPairs = cc.metadata().size();
    using PhiHisto = PhiBinner<TrackerTraits>;
    // uint32_t const* __restrict__ offsets = hh.hitsLayerStart().data();
    ALPAKA_ASSERT_ACC(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should much be bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    auto& innerLayerCumulativeSize = alpaka::declareSharedVar<uint32_t[TrackerTraits::nPairs], __COUNTER__>(acc);
    auto& ntot = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

    constexpr uint32_t dimIndexY = 0u;
    constexpr uint32_t dimIndexX = 1u;
    const uint32_t threadIdxLocalY(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndexY]);
    const uint32_t threadIdxLocalX(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[dimIndexX]);

    if (threadIdxLocalY == 0 && threadIdxLocalX == 0) {
      innerLayerCumulativeSize[0] = layerSize(cc.graph()[0][0]);//TrackerTraits::layerPairs[0]);
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

      uint8_t inner = TrackerTraits::layerPairs[2 * pairLayerId];
      uint8_t outer = TrackerTraits::layerPairs[2 * pairLayerId + 1];
      ALPAKA_ASSERT_ACC(outer > inner);

      auto hoff = PhiHisto::histOff(outer);
      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      ALPAKA_ASSERT_ACC(i >= offsets[inner]);
      ALPAKA_ASSERT_ACC(i < offsets[inner + 1]);

      // found hit corresponding to our worker thread, now do the job
      if (hh[i].detectorIndex() > pixelClustering::maxNumModules)
        continue;  // invalid

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto mez = hh[i].zGlobal();

      if (mez < cc.minz()[pairLayerId] || mez > cc.maxz()[pairLayerId])
        continue;

      if (cc.doClusterCut() && outer > pixelTopology::last_barrel_layer && clusterCut<TrackerTraits,TAcc>(acc, hh, i))
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
        return dr > cc.maxr()[pairLayerId] || dr < 0 || std::abs((mez * ro - mer * zo)) > cc.cellZ0Cut() * dr;
      };

      auto iphicut = cc.phiCuts()[pairLayerId];

      auto kl = PhiHisto::bin(int16_t(mep - iphicut));
      auto kh = PhiHisto::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % PhiHisto::nbins(); };

#ifdef GPU_DEBUG
      int tot = 0;
      int nmin = 0;
      int tooMany = 0;
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += phiBinner->size(kk + hoff);
#endif
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
          if (mo > pixelClustering::maxNumModules)
            continue;

          if (cc.cellZ0Cut() > 0 && z0cutoff(oi))
            continue;

          auto mop = hh[oi].iphi();
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          if (idphi > iphicut)
            continue;

          if (cc.doClusterCut() && zSizeCut<TrackerTraits,TAcc>(acc, hh, i, oi))
            continue;

          if (cc.cellPtCut() > 0 && ptcut(oi, idphi))
            continue;

          auto ind = alpaka::atomicAdd(acc, nCells, (uint32_t)1, alpaka::hierarchy::Blocks{});
          if (ind >= maxNumOfDoublets) {
            alpaka::atomicSub(acc, nCells, (uint32_t)1, alpaka::hierarchy::Blocks{});
            break;
          }  // move to SimpleVector??
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, inner, outer, i, oi);
          isOuterHitOfCell[oi].push_back(acc, ind);
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full())
            ++tooMany;
          ++tot;
#endif
        }
      }
//      #endif
#ifdef GPU_DEBUG
      if (tooMany > 0 or tot > 0)
        printf("OuterHitOfCell for %d in layer %d/%d, %d,%d %d, %d %.3f %.3f %s\n",
               i,
               inner,
               outer,
               nmin,
               tot,
               tooMany,
               iphicut,
               TrackerTraits::minz[pairLayerId],
               TrackerTraits::maxz[pairLayerId],
               tooMany > 0 ? "FULL!!" : "not full.");
#endif
    }  // loop in block...
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h

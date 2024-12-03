#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CACell.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {

  using namespace ::caStructures;
  template <typename TrackerTraits>
  using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracks = caStructures::CellTracksT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

  using HitToCell = caStructures::GenericContainer;

  template <typename TrackerTraits>
  class CAFishbone {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  HitsConstView hh,
                                  CACellT<TrackerTraits>* cells,
                                  uint32_t const* __restrict__ nCells,
                                  OuterHitOfCell<TrackerTraits> const* isOuterHitOfCellWrap,
                                  HitToCell const* __restrict__ outerHitHisto,
                                  uint32_t nHits,
                                  uint32_t nHitsBPix1,
                                  bool checkTrack) const {
      constexpr auto maxCellsPerHit = CACellT<TrackerTraits>::maxCellsPerHit;

      auto const isOuterHitOfCell = isOuterHitOfCellWrap->container;

      float x[maxCellsPerHit], y[maxCellsPerHit], z[maxCellsPerHit], n[maxCellsPerHit];
      uint32_t cc[maxCellsPerHit];
      uint16_t d[maxCellsPerHit];
      uint8_t l[maxCellsPerHit];

      // outermost parallel loop, using all grid elements along the slower dimension (Y or 0 in a 2D grid)
      for (uint32_t idy : cms::alpakatools::uniform_elements_y(acc, nHits - nHitsBPix1)) {
        auto const& vc = isOuterHitOfCell[idy];
        auto id = idy+nHitsBPix1;  //TODO have this offset in the histo building directly
        uint32_t histSize = outerHitHisto->size(idy+nHitsBPix1);
        auto size = vc.size();
        printf("hist %d histSize %d size %d \n",idy+nHitsBPix1,histSize,size);
        ALPAKA_ASSERT_ACC(histSize == uint32_t(size));
        if (size < 2)
          continue;
        // if alligned kill one of the two.
        // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
        auto const& c0 = cells[vc[0]];
        auto xo = c0.outer_x(hh);
        auto yo = c0.outer_y(hh);
        auto zo = c0.outer_z(hh);
        auto sg = 0;
        // this could be moved below 
        // precomputing these here has 
        // the advantage we then loop on less 
        // entries but we can anyway skip them below and avoid having 
        // the arrays above

        auto const* __restrict__ bin = outerHitHisto->begin(idy+nHitsBPix1);
        for (auto idx = 0u; idx < histSize; idx++) {
          unsigned int otherCell = bin[idx];
          printf("vc[0] %d idx %d vc[idx] %d otherCell %d \n",vc[0],idx,vc[idx],otherCell);
        }
        
        for (auto idx = 0u; idx < histSize; idx++) {
        // for (int32_t ic = 0; ic < size; ++ic) {
        // for (auto ic = 0u; ic < histSize; ic++) {
          unsigned int otherCell = bin[idx];
          auto& ci = cells[otherCell];//vc[ic]];
          // unsigned int otherCell = bin[ic] - nHitsBPix1;
          // auto& ci = cells[otherCell];
          if (ci.unused())
            continue;  // for triplets equivalent to next
          if (checkTrack && ci.tracks().empty())
            continue;
          cc[sg] = otherCell;//vc[ic];
          l[sg] = ci.layerPairId();
          d[sg] = ci.inner_detIndex(hh);
          x[sg] = ci.inner_x(hh) - xo;
          y[sg] = ci.inner_y(hh) - yo;
          z[sg] = ci.inner_z(hh) - zo;
          n[sg] = x[sg] * x[sg] + y[sg] * y[sg] + z[sg] * z[sg];
          ++sg;
        }
        if (sg < 2)
          continue;

        // innermost parallel loop, using the block elements along the faster dimension (X or 1 in a 2D grid)
        for (uint32_t ic : cms::alpakatools::independent_group_elements_x(acc, sg - 1)) {
          auto& ci = cells[cc[ic]];
          for (auto jc = ic + 1; (int)jc < sg; ++jc) {
            auto& cj = cells[cc[jc]];
            // must be different detectors (in the same layer)
            // if (d[ic]==d[jc]) continue;
            auto cos12 = x[ic] * x[jc] + y[ic] * y[jc] + z[ic] * z[jc];

            // cos12 * cos12 could go after d[ic] != d[jc]
            if (d[ic] != d[jc] && cos12 * cos12 >= 0.99999f * (n[ic] * n[jc])) {
              // alligned:  kill farthest (prefer consecutive layers)
              // if same layer prefer farthest (longer level arm) and make space for intermediate hit
              bool sameLayer = l[ic] == l[jc];
              if (n[ic] > n[jc]) {
                if (sameLayer) {
                  cj.kill();  // closest
                  ci.setFishbone(acc, cj.inner_hit_id(), cj.inner_z(hh), hh);
                } else {
                  ci.kill();  // farthest
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              } else {
                if (!sameLayer) {
                  cj.kill();  // farthest
                } else {
                  ci.kill();  // closest
                  cj.setFishbone(acc, ci.inner_hit_id(), ci.inner_z(hh), hh);
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              }
            }
          }  // cj
        }  // ci
      }  // hits
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h


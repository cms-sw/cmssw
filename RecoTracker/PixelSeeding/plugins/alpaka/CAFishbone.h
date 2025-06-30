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
  using CellToTracks = caStructures::GenericContainer;

  template <typename TrackerTraits>
  class CAFishbone {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  HitsConstView hh,
                                  CACell<TrackerTraits>* cells,
                                  uint32_t const* __restrict__ nCells,
                                  // OuterHitOfCell<TrackerTraits> const* isOuterHitOfCellWrap,
                                  HitToCell const* __restrict__ outerHitHisto,
                                  CellToTracks const* __restrict__ cellTracksHisto,
                                  uint32_t outerHits,
                                  bool checkTrack) const {
      constexpr auto maxCellsPerHit = TrackerTraits::maxCellsPerHit;

      // auto const isOuterHitOfCell = isOuterHitOfCellWrap->container;

      float x[maxCellsPerHit], y[maxCellsPerHit], z[maxCellsPerHit], n[maxCellsPerHit];
      uint32_t cc[maxCellsPerHit];
      uint16_t d[maxCellsPerHit];
      uint8_t l[maxCellsPerHit];

      // outermost parallel loop, using all grid elements along the slower dimension (Y or 0 in a 2D grid)
      for (uint32_t idy : cms::alpakatools::uniform_elements_y(acc, outerHits)) {
        // auto const& vc = isOuterHitOfCell[idy];
        uint32_t size = outerHitHisto->size(idy); //TODO have this offset in the histo building directly
// #ifdef GPU_DEBUG
//         printf("hist %d histSize %d \n",idy,size);
// #endif
        // printf("fishbone ---> outerhit %d size %d - ",idy,size);

        if (size < 2)
          continue;
        
        auto const* __restrict__ bin = outerHitHisto->begin(idy);
        auto const* __restrict__ end = outerHitHisto->end(idy);
        auto const nInBin = end - bin;

        // if alligned kill one of the two.
        // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
        auto const& c0 = cells[bin[0]];
        auto xo = c0.outer_x(hh);
        auto yo = c0.outer_y(hh);
        auto zo = c0.outer_z(hh);
        auto sg = 0;
        //printf("first cell %d x0 %.2f y0 %.2f z0 %.2f - ",bin[0],c0.outer_x(hh),c0.outer_y(hh),c0.outer_z(hh));

        // this could be moved below 
        // precomputing these here has 
        // the advantage we then loop on less 
        // entries but we can anyway skip them below and avoid having 
        // the arrays above

// #ifdef GPU_DEBUG 
//         for (auto idx = 0u; idx < size; idx++) {
//           unsigned int otherCell = bin[idx];
//           printf("vc[0] %d idx %d vc[idx] %d otherCell %d \n",vc[0],idx,vc[idx],otherCell);
//         }
// #endif
        for (auto idx = 0u; idx < nInBin; idx++) {
        // for (int32_t ic = 0; ic < size; ++ic) {
        // for (auto ic = 0u; ic < size; ic++) {
          unsigned int otherCell = bin[idx];
          auto& ci = cells[otherCell];//vc[ic]];
          // unsigned int otherCell = bin[ic] - nHitsBPix1;
          // auto& ci = cells[otherCell];
          if (ci.unused())
            continue;  // for triplets equivalent to next
          if (checkTrack && cellTracksHisto->size(otherCell) == 0)//ci.tracks().empty())
            continue;
          cc[sg] = otherCell;//vc[ic];
          l[sg] = ci.layerPairId();
          d[sg] = ci.inner_detIndex(hh);
          x[sg] = ci.inner_x(hh) - xo;
          y[sg] = ci.inner_y(hh) - yo;
          z[sg] = ci.inner_z(hh) - zo;
          n[sg] = x[sg] * x[sg] + y[sg] * y[sg] + z[sg] * z[sg];
          ++sg;
          //printf("sg %d idx %d cell %d x %.2f y %.2f z %.2f - ",sg,idx,otherCell,x[sg],y[sg],z[sg]);
        }
        //printf("\n");
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
#ifdef GPU_DEBUG
printf("hit %d same layer cell %d kill %d \n",idy,cc[ic],cc[jc]);  
#endif
                  ci.setFishbone(acc, cj.inner_hit_id(), cj.inner_z(hh), hh);
                } else {
                  ci.kill();  // farthest
#ifdef GPU_DEBUG
printf("hit %d same layer cell %d kill %d \n",idy,cc[jc],cc[ic]);  
#endif
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              } else {
                if (!sameLayer) {
                  cj.kill();  // farthest
#ifdef GPU_DEBUG
printf("hit %d diff layer cell %d kill %d \n",idy,cc[ic],cc[jc]);
#endif  
                } else {
                  ci.kill();  // closest
#ifdef GPU_DEBUG
printf("hit %d diff layer cell %d kill %d \n",idy,cc[jc],cc[ic]);
#endif
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


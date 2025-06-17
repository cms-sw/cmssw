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

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {

  using HitToCell = caStructures::GenericContainer;
  using CellToTracks = caStructures::GenericContainer;

  template <typename TrackerTraits>
  class CAFishbone {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  HitsConstView hh,
                                  CACell<TrackerTraits>* cells,
                                  uint32_t const* __restrict__ nCells,
                                  HitToCell const* __restrict__ outerHitHisto,
                                  CellToTracks const* __restrict__ cellTracksHisto,
                                  uint32_t outerHits,
                                  bool checkTrack) const {
      // outermost parallel loop, using all grid elements along the slower dimension (Y or 0 in a 2D grid)
      for (uint32_t idy : cms::alpakatools::uniform_elements_y(acc, outerHits)) {
        uint32_t size = outerHitHisto->size(idy);
#ifdef GPU_DEBUG
        printf("fishbone ---> outersize %d - ", idy, size);
#endif
        if (size < 2)
          continue;

        auto const* __restrict__ bin = outerHitHisto->begin(idy);

        // if alligned kill one of the two.
        // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
        auto const& c0 = cells[bin[0]];
        auto xo = c0.outer_x(hh);
        auto yo = c0.outer_y(hh);
        auto zo = c0.outer_z(hh);
        //printf("first cell %d xo %.2f yo %.2f zo %.2f - ",bin[0],c0.outer_x(hh),c0.outer_y(hh),c0.outer_z(hh));ve

#ifdef GPU_DEBUG
        for (auto idx = 0u; idx < size; idx++) {
          unsigned int otherCell = bin[idx];
          printf("vc[0] %d idx %d vc[idx] %d otherCell %d \n", vc[0], idx, vc[idx], otherCell);
        }
#endif
        for (uint32_t ic : cms::alpakatools::independent_group_elements_x(acc, size)) {
          //printf("cell0 = %d ci = %d\n",bin[0],bin[ic]);
          unsigned int otherCell = bin[ic];
          auto& ci = cells[otherCell];
          //	printf("xo = %.2f yo = %.2f zo = %.2f xi = %.2f yi = %.2f zi = %.2f \n",xo,yo,zo,ci.inner_x(hh),ci.inner_y(hh),ci.inner_z(hh));
          if (ci.unused())
            continue;  // for triplets equivalent to next
          if (checkTrack && cellTracksHisto->size(otherCell) == 0)
            continue;

          float x1 = (ci.inner_x(hh) - xo);
          float y1 = (ci.inner_y(hh) - yo);
          float z1 = (ci.inner_z(hh) - zo);
          float n1 = x1 * x1 + y1 * y1 + z1 * z1;

          for (auto jc = ic + 1; jc < size; ++jc) {
            unsigned int nextCell = bin[jc];
            auto& cj = cells[nextCell];
            if (cj.unused())
              continue;
            if (checkTrack && cellTracksHisto->size(nextCell) == 0)
              continue;
#ifdef GPU_DEBUG
            printf("xx = %.2f yo = %.2f zo = %.2f xi = %.2f yi = %.2f zi = %.2f xj = %.2f yj = %.2f zj = %.2f\n",
                   xo,
                   yo,
                   zo,
                   ci.inner_x(hh),
                   ci.inner_y(hh),
                   ci.inner_z(hh),
                   cj.inner_x(hh),
                   cj.inner_y(hh),
                   cj.inner_z(hh));
#endif

            if (ci.inner_detIndex(hh) == cj.inner_detIndex(hh))
              continue;

            float x2 = (cj.inner_x(hh) - xo);
            float y2 = (cj.inner_y(hh) - yo);
            float z2 = (cj.inner_z(hh) - zo);
            float n2 = x2 * x2 + y2 * y2 + z2 * z2;

            auto cos12 = x1 * x2 + y1 * y2 + z1 * z2;

            if (cos12 * cos12 >= 0.99999f * (n1 * n2)) {
              // alligned:  kill farthest (prefer consecutive layers)
              // if same layer prefer farthest (longer level arm) and make space for intermediate hit
              bool sameLayer = int(ci.layerPairId()) == int(cj.layerPairId());
              if (n1 > n2) {
                if (sameLayer) {
                  cj.kill();  // closest
                  ci.setFishbone(acc, cj.inner_hit_id(), cj.inner_z(hh), hh);
#ifdef GPU_DEBUG
                  printf(
                      "n1>n2 lic = %d ljc = %d dic = %.2f djc = %.2f cell %d kill %d cos = %.7f n1 = %.3f n2 = %.3f "
                      "same\n",
                      int(ci.layerPairId()),
                      int(cj.layerPairId()),
                      ci.inner_detIndex(hh),
                      cj.inner_detIndex(hh),
                      bin[ic],
                      bin[jc],
                      cos12 * cos12 / (n1 * n2),
                      n1,
                      n2);
#endif
                } else {
                  ci.kill();  // farthest
#ifdef GPU_DEBUG
                  printf(
                      "n1>n2 lic = %d ljc = %d dic = %.2f djc = %.2f cell %d kill %d cos = %.7f n1 = %.3f n2 = %.3f "
                      "diff\n",
                      int(ci.layerPairId()),
                      int(cj.layerPairId()),
                      ci.inner_detIndex(hh),
                      cj.inner_detIndex(hh),
                      bin[jc],
                      bin[ic],
                      cos12 * cos12 / (n1 * n2),
                      n1,
                      n2);
#endif
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              } else {
                if (!sameLayer) {
                  cj.kill();  // farthest
#ifdef GPU_DEBUG
                  printf(
                      "n2>n1 lic = %d ljc = %d dic = %.2f djc = %.2f cell %d kill %d cos = %.7f n1 = %.3f n2 = %.3f "
                      "diff\n",
                      int(ci.layerPairId()),
                      int(cj.layerPairId()),
                      ci.inner_detIndex(hh),
                      cj.inner_detIndex(hh),
                      bin[ic],
                      bin[jc],
                      cos12 * cos12 / (n1 * n2),
                      n1,
                      n2);
#endif
                } else {
                  ci.kill();  // closest
                  cj.setFishbone(acc, ci.inner_hit_id(), ci.inner_z(hh), hh);
#ifdef GPU_DEBUG
                  printf(
                      "n2>n1 lic = %d ljc = %d dic = %.2f djc = %.2f cell %d kill %d cos = %.7f n1 = %.3f n2 = %.3f "
                      "same\n",
                      int(ci.layerPairId()),
                      int(cj.layerPairId()),
                      ci.inner_detIndex(hh),
                      cj.inner_detIndex(hh),
                      bin[jc],
                      bin[ic],
                      cos12 * cos12 / (n1 * n2),
                      n1,
                      n2);
#endif
                  // break;  // removed to improve reproducibility, keep it for reference and tests
                }
              }
            }  // cos
          }  // cj
        }  // ci
      }  // hits
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAFishbone_h

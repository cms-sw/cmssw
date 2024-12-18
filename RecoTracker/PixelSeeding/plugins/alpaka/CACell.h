#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CACell_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

// #define ONLY_TRIPLETS_IN_HOLE

#include <cmath>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/CACoupleSoA.h"

#include "CAStructures.h"
#include "CASimpleCell.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::caStructures;

  template <typename TrackerTraits>
  class CACellT : public CASimpleCell<TrackerTraits> {
  public:

    using typename CASimpleCell<TrackerTraits>::StatusBit;
    using typename CASimpleCell<TrackerTraits>::Quality;
    using typename CASimpleCell<TrackerTraits>::HitContainer;
    using typename CASimpleCell<TrackerTraits>::TmpTuple;

    using PtrAsInt = unsigned long long;

    using OuterHitOfCellContainer = OuterHitOfCellContainerT<TrackerTraits>;
    using OuterHitOfCell = OuterHitOfCellT<TrackerTraits>;
    using CellNeighbors = CellNeighborsT<TrackerTraits>;
    using CellTracks = CellTracksT<TrackerTraits>;
    using CellNeighborsVector = CellNeighborsVectorT<TrackerTraits>;
    using CellTracksVector = CellTracksVectorT<TrackerTraits>;

    CACellT() = default;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(CellNeighborsVector& cellNeighbors,
                                             CellTracksVector& cellTracks,
                                             const HitsConstView& hh,
                                             int layerPairId,
                                             uint8_t theInnerLayer,
                                             uint8_t theOuterLayer,
                                             hindex_type innerHitId,
                                             hindex_type outerHitId) {
      this->theInnerHitId = innerHitId;
      this->theOuterHitId = outerHitId;
      this->theLayerPairId_ = layerPairId;
      this->theInnerLayer_ = theInnerLayer;
      this->theOuterLayer_ = theOuterLayer;
      this->theStatus_ = 0;
      this->theFishboneId = this->invalidHitId;

      // optimization that depends on access pattern
      this->theInnerZ = hh[innerHitId].zGlobal();
      this->theInnerR = hh[innerHitId].rGlobal();

      // link to default empty
      theOuterNeighbors = &cellNeighbors[0];
      theTracks = &cellTracks[0];
      ALPAKA_ASSERT_ACC(outerNeighbors().empty());
      ALPAKA_ASSERT_ACC(tracks().empty());
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) int addOuterNeighbor(
        const TAcc& acc, typename TrackerTraits::cindex_type t, CellNeighborsVector& cellNeighbors) {
      // use smart cache
      if (outerNeighbors().empty()) {
        auto i = cellNeighbors.extend(acc);  // maybe wasted....
        if (i > 0) {
          cellNeighbors[i].reset();
          alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          theOuterNeighbors = &cellNeighbors[i];
#else
          auto zero = (PtrAsInt)(&cellNeighbors[0]);
          alpaka::atomicCas(acc,
                            (PtrAsInt*)(&theOuterNeighbors),
                            zero,
                            (PtrAsInt)(&cellNeighbors[i]),
                            alpaka::hierarchy::Blocks{});  // if fails we cannot give "i" back...
#endif
        } else
          return -1;
      }
      alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
      return outerNeighbors().push_back(acc, t);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) int addTrack(TAcc const& acc,
                                                                               tindex_type t,
                                                                               CellTracksVector& cellTracks) {
      if (tracks().empty()) {
        auto i = cellTracks.extend(acc);  // maybe wasted....
        if (i > 0) {
          cellTracks[i].reset();
          alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          theTracks = &cellTracks[i];
#else
          auto zero = (PtrAsInt)(&cellTracks[0]);
          alpaka::atomicCas(acc,
                            (PtrAsInt*)(&theTracks),
                            zero,
                            (PtrAsInt)(&cellTracks[i]),
                            alpaka::hierarchy::Blocks{});  // if fails we cannot give "i" back...

#endif
        } else
          return -1;
      }
      alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
      return tracks().push_back(acc, t);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE CellTracks& tracks() { return *theTracks; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE CellTracks const& tracks() const { return *theTracks; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE CellNeighbors& outerNeighbors() { return *theOuterNeighbors; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE CellNeighbors const& outerNeighbors() const { return *theOuterNeighbors; }

    ALPAKA_FN_ACC bool check_alignment(const HitsConstView& hh,
                                       CACellT const& otherCell,
                                       const float ptmin,
                                       const float hardCurvCut,
                                       const float caThetaCutBarrel,
                                       const float caThetaCutForward,
                                       const float dcaCutInnerTriplet,
                                       const float dcaCutOuterTriplet) const {
      // detIndex of the layerStart for the Phase1 Pixel Detector:
      // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
      // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
      auto ri = this->inner_r(hh);
      auto zi = this->inner_z(hh);

      auto ro = this->outer_r(hh);
      auto zo = this->outer_z(hh);

      auto r1 = otherCell.inner_r(hh);
      auto z1 = otherCell.inner_z(hh);
      auto isBarrel = otherCell.outer_detIndex(hh) < TrackerTraits::last_barrel_detIndex;
      // TODO tune CA cuts below (theta and dca)
      bool aligned = areAlignedRZ(r1, z1, ri, zi, ro, zo, ptmin, isBarrel ? caThetaCutBarrel : caThetaCutForward);
      return (aligned && dcaCut(hh,
                                otherCell,
                                otherCell.inner_detIndex(hh) < TrackerTraits::last_bpix1_detIndex ? dcaCutInnerTriplet
                                                                                                  : dcaCutOuterTriplet,
                                hardCurvCut));
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) static bool dcaCutH(
        float x1,
        float y1,
        float x2,
        float y2,
        float x3,
        float y3,
        const float region_origin_radius_plus_tolerance,
        const float maxCurv) {
      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

      if (eq.curvature() > maxCurv)
        return false;

      return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
    }


#ifdef CA_TRIPLET_HOLES

    // These functions have never been used in production
    // They need an AverageGeometry to be filled 
    // Commenting for the moment since they are the only reason we 
    // fill the AverageGeometry and attach to the hit SoA

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hole0(const HitsConstView& hh, AverageGeometryConstView& ag, CACellT const& innerCell) const {
      using namespace phase1PixelTopology;

      int p = innerCell.inner_iphi(hh);
      if (p < 0)
        p += std::numeric_limits<unsigned short>::max();
      p = (max_ladder_bpx0 * p) / std::numeric_limits<unsigned short>::max();
      p %= max_ladder_bpx0;
      auto il = first_ladder_bpx0 + p;
      auto r0 = ag[il].ladderR();
      auto ri = innerCell.inner_r(hh);
      auto zi = innerCell.inner_z(hh);
      auto ro = this->outer_r(hh);
      auto zo = this->outer_z(hh);
      auto z0 = zi + (r0 - ri) * (zo - zi) / (ro - ri);
      auto z_in_ladder = std::abs(z0 - ag[il].ladderZ());
      auto z_in_module = z_in_ladder - module_length_bpx0 * int(z_in_ladder / module_length_bpx0);
      auto gap = z_in_module < module_tolerance_bpx0 || z_in_module > (module_length_bpx0 - module_tolerance_bpx0);
      return gap;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hole4(const HitsConstView& hh, CACellT const& innerCell) const {
      using namespace phase1PixelTopology;

      int p = this->outer_iphi(hh);
      if (p < 0)
        p += std::numeric_limits<unsigned short>::max();
      p = (max_ladder_bpx4 * p) / std::numeric_limits<unsigned short>::max();
      p %= max_ladder_bpx4;
      auto il = first_ladder_bpx4 + p;
      auto r4 = ag[il].ladderR();
      auto ri = innerCell.inner_r(hh);
      auto zi = innerCell.inner_z(hh);
      auto ro = this->outer_r(hh);
      auto zo = this->outer_z(hh);
      auto z4 = zo + (r4 - ro) * (zo - zi) / (ro - ri);
      auto z_in_ladder = std::abs(z4 - ag[il].ladderZ());
      auto z_in_module = z_in_ladder - module_length_bpx4 * int(z_in_ladder / module_length_bpx4);
      auto gap = z_in_module < module_tolerance_bpx4 || z_in_module > (module_length_bpx4 - module_tolerance_bpx4);
      auto holeP = z4 > ag[il].ladderMaxZ() && z4 < ag[0].endCapZ();
      auto holeN = z4 < ag[il].ladderMinZ() && z4 > ag[1].endCapZ();
      return gap || holeP || holeN;
    }
#endif

    
    // trying to free the track building process from hardcoded layers, leaving
    // the visit of the graph based on the neighborhood connections between cells.
    template <int DEPTH, typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void find_ntuplets(TAcc const& acc,
                                                      const HitsConstView& hh,
                                                      CACellT* __restrict__ cells,
                                                      CellTracksVector& cellTracks,
                                                      HitContainer& foundNtuplets,
                                                      cms::alpakatools::AtomicPairCounter& apc,
                                                      Quality* __restrict__ quality,
                                                      TmpTuple& tmpNtuplet,
                                                      const unsigned int minHitsPerNtuplet,
                                                      bool startAt0) const {
      // the building process for a track ends if:
      // it has no right neighbor
      // it has no compatible neighbor
      // the ntuplets is then saved if the number of hits it contains is greater
      // than a threshold

      if constexpr (DEPTH <= 0) {
        printf("ERROR: CACellT::find_ntuplets reached full depth!\n");
        ALPAKA_ASSERT_ACC(false);
      } else {
        auto doubletId = this - cells;
        tmpNtuplet.push_back_unsafe(doubletId);
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() <= int(TrackerTraits::maxHitsOnTrack - 3));

        bool last = true;
        for (unsigned int otherCell : outerNeighbors()) {
          if (cells[otherCell].isKilled())
            continue;  // killed by earlyFishbone
          last = false;
          cells[otherCell].template find_ntuplets<DEPTH - 1>(
              acc, hh, cells, cellTracks, foundNtuplets, apc, quality, tmpNtuplet, minHitsPerNtuplet, startAt0);
        }
        if (last) {  // if long enough save...
          if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
#ifdef ONLY_TRIPLETS_IN_HOLE
            // triplets accepted only pointing to the hole
            if (tmpNtuplet.size() >= 3 || (startAt0 && hole4(hh, cells[tmpNtuplet[0]])) ||
                ((!startAt0) && hole0(hh, cells[tmpNtuplet[0]])))
#endif
            {
              hindex_type hits[TrackerTraits::maxDepth + 2];
              auto nh = 0U;
              constexpr int maxFB = 2;  // for the time being let's limit this
              int nfb = 0;
              for (auto c : tmpNtuplet) {
                hits[nh++] = cells[c].theInnerHitId;
                if (nfb < maxFB && cells[c].hasFishbone()) {
                  ++nfb;
                  hits[nh++] = cells[c].theFishboneId;  // Fishbone hit is always outer than inner hit
                }
              }
              ALPAKA_ASSERT_ACC(nh < TrackerTraits::maxHitsOnTrack);
              hits[nh] = this->theOuterHitId;
              auto it = foundNtuplets.bulkFill(acc, apc, hits, nh + 1);
              if (it >= 0) {  // if negative is overflow....
                for (auto c : tmpNtuplet)
                  cells[c].addTrack(acc, it, cellTracks);
                quality[it] = this->bad;  // initialize to bad
              }
            }
          }
        }
        tmpNtuplet.pop_back();
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() < int(TrackerTraits::maxHitsOnTrack - 1));
      }
    }

  private:

    CellNeighbors* theOuterNeighbors;
    CellTracks* theTracks;

  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

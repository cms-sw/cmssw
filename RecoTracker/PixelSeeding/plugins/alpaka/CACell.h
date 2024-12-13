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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::caStructures;

  template <typename TrackerTraits>
  class CASimpleCell {
  
   public:
   
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(const HitsConstView& hh,
                                             int layerPairId,
                                             uint8_t theInnerLayer,
                                             uint8_t theOuterLayer,
                                             hindex_type innerHitId,
                                             hindex_type outerHitId) {
      theInnerHitId = innerHitId;
      theOuterHitId = outerHitId;
      theLayerPairId_ = layerPairId;
      theInnerLayer_ = theInnerLayer;
      theOuterLayer_ = theOuterLayer;
      theStatus_ = 0;
      theFishboneId = invalidHitId;

      // optimization that depends on access pattern
      theInnerZ = hh[innerHitId].zGlobal();
      theInnerR = hh[innerHitId].rGlobal();
    }

    using hindex_type = typename TrackerTraits::hindex_type;
    using tindex_type = typename TrackerTraits::tindex_type;
    static constexpr auto invalidHitId = std::numeric_limits<hindex_type>::max();

    using TmpTuple = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxDepth>;
    using HitContainer = caStructures::SequentialContainer;
    using CellToCell = caStructures::GenericContainer;
    using CellToTracks = caStructures::GenericContainer;
    using CACoupleSoAView = caStructures::CACoupleSoAView;

    using Quality = ::pixelTrack::Quality;
    static constexpr auto bad = ::pixelTrack::Quality::bad;

    enum class StatusBit : uint16_t { kUsed = 1, kInTrack = 2, kKilled = 1 << 15 };

    CASimpleCell() = default;
    
    constexpr unsigned int inner_hit_id() const { return theInnerHitId; }
    constexpr unsigned int outer_hit_id() const { return theOuterHitId; }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void kill() { theStatus_ |= uint16_t(StatusBit::kKilled); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isKilled() const { return theStatus_ & uint16_t(StatusBit::kKilled); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t layerPairId() const { return theLayerPairId_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t innerLayer() const { return theInnerLayer_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t outerLayer() const { return theOuterLayer_; }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool unused() const { return 0 == (uint16_t(StatusBit::kUsed) & theStatus_); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setStatusBits(StatusBit mask) { theStatus_ |= uint16_t(mask); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_x(const HitsConstView& hh) const { return hh[theInnerHitId].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_x(const HitsConstView& hh) const { return hh[theOuterHitId].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_y(const HitsConstView& hh) const { return hh[theInnerHitId].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_y(const HitsConstView& hh) const { return hh[theOuterHitId].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_z(const HitsConstView& hh) const { return theInnerZ; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_z(const HitsConstView& hh) const { return hh[theOuterHitId].zGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_r(const HitsConstView& hh) const { return theInnerR; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_r(const HitsConstView& hh) const { return hh[theOuterHitId].rGlobal(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_iphi(const HitsConstView& hh) const { return hh[theInnerHitId].iphi(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto outer_iphi(const HitsConstView& hh) const { return hh[theOuterHitId].iphi(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_detIndex(const HitsConstView& hh) const {
      return hh[theInnerHitId].detectorIndex();
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_detIndex(const HitsConstView& hh) const {
      return hh[theOuterHitId].detectorIndex();
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto fishboneId() const { return theFishboneId; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hasFishbone() const { return theFishboneId != invalidHitId; }


    ALPAKA_FN_ACC void print_cell() const {
      printf("printing cell: on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
             theLayerPairId_,
             theInnerHitId,
             theOuterHitId);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setFishbone(TAcc const& acc, hindex_type id, float z, const HitsConstView& hh) {
      // make it deterministic: use the farther apart (in z)
      auto old = theFishboneId;
      while (old !=
             alpaka::atomicCas(
                 acc,
                 &theFishboneId,
                 old,
                 (invalidHitId == old || std::abs(z - theInnerZ) > std::abs(hh[old].zGlobal() - theInnerZ)) ? id : old,
                 alpaka::hierarchy::Blocks{}))
        old = theFishboneId;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) static bool areAlignedRZ(
        float r1, float z1, float ri, float zi, float ro, float zo, const float ptmin, const float thetaCut) {
      float radius_diff = std::abs(r1 - ro);
      float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

      float pMin = ptmin * std::sqrt(distance_13_squared);  // this needs to be divided by
                                                            // radius_diff later

      float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
      return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool dcaCut(const HitsConstView& hh,
                                               CASimpleCell const& otherCell,
                                               const float region_origin_radius_plus_tolerance,
                                               const float maxCurv) const {
      auto x1 = otherCell.inner_x(hh);
      auto y1 = otherCell.inner_y(hh);

      auto x2 = inner_x(hh);
      auto y2 = inner_y(hh);

      auto x3 = outer_x(hh);
      auto y3 = outer_y(hh);

      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

      if (eq.curvature() > maxCurv)
        return false;

      return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
    }

    // trying to free the track building process from hardcoded layers, leaving
    // the visit of the graph based on the neighborhood connections between cells.
    template <int DEPTH, typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void find_ntuplets(TAcc const& acc,
                                                      const HitsConstView& hh,
                                                      const ::reco::CAGraphSoAConstView &cc,
                                                      CASimpleCell* __restrict__ cells,
                                                      HitContainer& foundNtuplets,
                                                      CellToCell const *__restrict__ cellNeighborsHisto,
                                                      CellToTracks *cellTracksHisto,
                                                      uint32_t *nCellTracks,
                                                      CACoupleSoAView ct,
                                                      cms::alpakatools::AtomicPairCounter& apc,
                                                      Quality* __restrict__ quality,
                                                      TmpTuple& tmpNtuplet,
                                                      const unsigned int minHitsPerNtuplet) const {
      // the building process for a track ends if:
      // it has no right neighbor
      // it has no compatible neighbor
      // the ntuplets is then saved if the number of hits it contains is greater
      // than a threshold

      if constexpr (DEPTH <= 0) {
        printf("ERROR: CASimpleCell::find_ntuplets reached full depth!\n");
        ALPAKA_ASSERT_ACC(false);
      } else {
        auto doubletId = this - cells;
        tmpNtuplet.push_back_unsafe(doubletId); // if we move this to be safe we could parallelize further below?
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() <= int(TrackerTraits::maxHitsOnTrack - 3));

        bool last = true;
        // for (auto o = cellNeighborsHisto->begin(doubletId); o != cellNeighborsHisto->end(doubletId); ++o)
        //  printf("doubletIdHisto: %ld -> %d\n",doubletId,*o);
        auto const* __restrict__ bin = cellNeighborsHisto->begin(doubletId);
        auto const* __restrict__ end = cellNeighborsHisto->end(doubletId);
        auto const nInBin = end - bin;

        for (auto idx = 0u; idx < nInBin; idx++) {
          // FIXME implement alpaka::ldg and use it here? or is it const* __restrict__ enough?
          unsigned int otherCell = bin[idx];
        // for (unsigned int otherCell : outerNeighbors()) {
#ifdef GPU_DEBUG
          printf("doubletId: %ld -> %d\n",doubletId,otherCell);
#endif
          if (cells[otherCell].isKilled())
            continue;  // killed by earlyFishbone
          last = false;
          cells[otherCell].template find_ntuplets<DEPTH - 1>(
              acc, hh, cc, cells, foundNtuplets, cellNeighborsHisto, cellTracksHisto, nCellTracks, ct, apc, quality, tmpNtuplet, minHitsPerNtuplet);
        }
        if (last) {  // if long enough save...
          if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
#ifdef ONLY_TRIPLETS_IN_HOLE
            // triplets accepted only pointing to the hole
            auto startAt0 = cc[this->layerPairId()].graph()[0] == 0;
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
              hits[nh] = theOuterHitId;
              auto it = foundNtuplets.bulkFill(acc, apc, hits, nh + 1);
#ifdef GPU_DEBUG
              printf("track n. %d nhits %d \n",it,nh+1);
#endif
              if (it >= 0) {  // if negative is overflow....
                for (auto c : tmpNtuplet)
                {

                  cellTracksHisto->count(acc,c); //use this to count!!!
                  auto t_ind = alpaka::atomicAdd(acc, nCellTracks, (uint32_t)1, alpaka::hierarchy::Blocks{});
                  // add a check
#ifdef GPU_DEBUG
                  printf("cellToTrack n. %d cell %d track %d\n",t_ind,c,it);
#endif
                  ct[t_ind].inner() = c; //cell
                  ct[t_ind].outer() = it; //track
                }
                quality[it] = bad;  // initialize to bad
              }
            }
          }
        }
        tmpNtuplet.pop_back();
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() < int(TrackerTraits::maxHitsOnTrack - 1));
      }
    }


    protected:
      int16_t theLayerPairId_;
      uint8_t theInnerLayer_;
      uint8_t theOuterLayer_;
      uint16_t theStatus_;  // tbd

      float theInnerZ;
      float theInnerR;
      hindex_type theInnerHitId;
      hindex_type theOuterHitId;
      hindex_type theFishboneId;

  };
  
  template <typename TrackerTraits>
  class CACellT {
  public:
    using PtrAsInt = unsigned long long;

    static constexpr auto maxCellsPerHit = TrackerTraits::maxCellsPerHit;
    using OuterHitOfCellContainer = OuterHitOfCellContainerT<TrackerTraits>;
    using OuterHitOfCell = OuterHitOfCellT<TrackerTraits>;
    using CellNeighbors = CellNeighborsT<TrackerTraits>;
    using CellTracks = CellTracksT<TrackerTraits>;
    using CellNeighborsVector = CellNeighborsVectorT<TrackerTraits>;
    using CellTracksVector = CellTracksVectorT<TrackerTraits>;

    using hindex_type = typename TrackerTraits::hindex_type;
    using tindex_type = typename TrackerTraits::tindex_type;
    static constexpr auto invalidHitId = std::numeric_limits<hindex_type>::max();

    using TmpTuple = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxDepth>;
    using HitContainer = caStructures::SequentialContainer;
    using CellToCell = caStructures::GenericContainer;
    using CellToTracks = caStructures::GenericContainer;

    using Quality = ::pixelTrack::Quality;
    static constexpr auto bad = ::pixelTrack::Quality::bad;

    enum class StatusBit : uint16_t { kUsed = 1, kInTrack = 2, kKilled = 1 << 15 };

    CACellT() = default;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(CellNeighborsVector& cellNeighbors,
                                             CellTracksVector& cellTracks,
                                             const HitsConstView& hh,
                                             int layerPairId,
                                             uint8_t theInnerLayer,
                                             uint8_t theOuterLayer,
                                             hindex_type innerHitId,
                                             hindex_type outerHitId) {
      theInnerHitId = innerHitId;
      theOuterHitId = outerHitId;
      theLayerPairId_ = layerPairId;
      theInnerLayer_ = theInnerLayer;
      theOuterLayer_ = theOuterLayer;
      theStatus_ = 0;
      theFishboneId = invalidHitId;

      // optimization that depends on access pattern
      theInnerZ = hh[innerHitId].zGlobal();
      theInnerR = hh[innerHitId].rGlobal();

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
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_x(const HitsConstView& hh) const { return hh[theInnerHitId].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_x(const HitsConstView& hh) const { return hh[theOuterHitId].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_y(const HitsConstView& hh) const { return hh[theInnerHitId].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_y(const HitsConstView& hh) const { return hh[theOuterHitId].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_z(const HitsConstView& hh) const { return theInnerZ; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_z(const HitsConstView& hh) const { return hh[theOuterHitId].zGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_r(const HitsConstView& hh) const { return theInnerR; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_r(const HitsConstView& hh) const { return hh[theOuterHitId].rGlobal(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_iphi(const HitsConstView& hh) const { return hh[theInnerHitId].iphi(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto outer_iphi(const HitsConstView& hh) const { return hh[theOuterHitId].iphi(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_detIndex(const HitsConstView& hh) const {
      return hh[theInnerHitId].detectorIndex();
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_detIndex(const HitsConstView& hh) const {
      return hh[theOuterHitId].detectorIndex();
    }

    constexpr unsigned int inner_hit_id() const { return theInnerHitId; }
    constexpr unsigned int outer_hit_id() const { return theOuterHitId; }

    ALPAKA_FN_ACC void print_cell() const {
      printf("printing cell: on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
             theLayerPairId_,
             theInnerHitId,
             theOuterHitId);
    }

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
      auto ri = inner_r(hh);
      auto zi = inner_z(hh);

      auto ro = outer_r(hh);
      auto zo = outer_z(hh);

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

    ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) static bool areAlignedRZ(
        float r1, float z1, float ri, float zi, float ro, float zo, const float ptmin, const float thetaCut) {
      float radius_diff = std::abs(r1 - ro);
      float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

      float pMin = ptmin * std::sqrt(distance_13_squared);  // this needs to be divided by
                                                            // radius_diff later

      float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
      return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool dcaCut(const HitsConstView& hh,
                                               CACellT const& otherCell,
                                               const float region_origin_radius_plus_tolerance,
                                               const float maxCurv) const {
      auto x1 = otherCell.inner_x(hh);
      auto y1 = otherCell.inner_y(hh);

      auto x2 = inner_x(hh);
      auto y2 = inner_y(hh);

      auto x3 = outer_x(hh);
      auto y3 = outer_y(hh);

      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

      if (eq.curvature() > maxCurv)
        return false;

      return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
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
      auto ro = outer_r(hh);
      auto zo = outer_z(hh);
      auto z0 = zi + (r0 - ri) * (zo - zi) / (ro - ri);
      auto z_in_ladder = std::abs(z0 - ag[il].ladderZ());
      auto z_in_module = z_in_ladder - module_length_bpx0 * int(z_in_ladder / module_length_bpx0);
      auto gap = z_in_module < module_tolerance_bpx0 || z_in_module > (module_length_bpx0 - module_tolerance_bpx0);
      return gap;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hole4(const HitsConstView& hh, CACellT const& innerCell) const {
      using namespace phase1PixelTopology;

      int p = outer_iphi(hh);
      if (p < 0)
        p += std::numeric_limits<unsigned short>::max();
      p = (max_ladder_bpx4 * p) / std::numeric_limits<unsigned short>::max();
      p %= max_ladder_bpx4;
      auto il = first_ladder_bpx4 + p;
      auto r4 = ag[il].ladderR();
      auto ri = innerCell.inner_r(hh);
      auto zi = innerCell.inner_z(hh);
      auto ro = outer_r(hh);
      auto zo = outer_z(hh);
      auto z4 = zo + (r4 - ro) * (zo - zi) / (ro - ri);
      auto z_in_ladder = std::abs(z4 - ag[il].ladderZ());
      auto z_in_module = z_in_ladder - module_length_bpx4 * int(z_in_ladder / module_length_bpx4);
      auto gap = z_in_module < module_tolerance_bpx4 || z_in_module > (module_length_bpx4 - module_tolerance_bpx4);
      auto holeP = z4 > ag[il].ladderMaxZ() && z4 < ag[0].endCapZ();
      auto holeN = z4 < ag[il].ladderMinZ() && z4 > ag[1].endCapZ();
      return gap || holeP || holeN;
    }
#endif

    // Cell status management
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void kill() { theStatus_ |= uint16_t(StatusBit::kKilled); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isKilled() const { return theStatus_ & uint16_t(StatusBit::kKilled); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t layerPairId() const { return theLayerPairId_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t innerLayer() const { return theInnerLayer_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t outerLayer() const { return theOuterLayer_; }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool unused() const { return 0 == (uint16_t(StatusBit::kUsed) & theStatus_); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setStatusBits(StatusBit mask) { theStatus_ |= uint16_t(mask); }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setFishbone(TAcc const& acc, hindex_type id, float z, const HitsConstView& hh) {
      // make it deterministic: use the farther apart (in z)
      auto old = theFishboneId;
      while (old !=
             alpaka::atomicCas(
                 acc,
                 &theFishboneId,
                 old,
                 (invalidHitId == old || std::abs(z - theInnerZ) > std::abs(hh[old].zGlobal() - theInnerZ)) ? id : old,
                 alpaka::hierarchy::Blocks{}))
        old = theFishboneId;
    }
    
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto fishboneId() const { return theFishboneId; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hasFishbone() const { return theFishboneId != invalidHitId; }
    
    // trying to free the track building process from hardcoded layers, leaving
    // the visit of the graph based on the neighborhood connections between cells.
    template <int DEPTH, typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void find_ntuplets(TAcc const& acc,
                                                      const HitsConstView& hh,
                                                      const ::reco::CAGraphSoAConstView &cc,
                                                      CACellT* __restrict__ cells,
                                                      CellTracksVector& cellTracks,
                                                      HitContainer& foundNtuplets,
                                                      CellToCell const *__restrict__ cellNeighborsHisto,
                                                      CellToTracks *cellTracksHisto,
                                                      cms::alpakatools::AtomicPairCounter& apc,
                                                      Quality* __restrict__ quality,
                                                      TmpTuple& tmpNtuplet,
                                                      const unsigned int minHitsPerNtuplet) const {
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
        tmpNtuplet.push_back_unsafe(doubletId); // if we move this to be safe we could parallelize further below?
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() <= int(TrackerTraits::maxHitsOnTrack - 3));

        bool last = true;
        // for (auto o = cellNeighborsHisto->begin(doubletId); o != cellNeighborsHisto->end(doubletId); ++o)
        //  printf("doubletIdHisto: %ld -> %d\n",doubletId,*o);
        auto const* __restrict__ bin = cellNeighborsHisto->begin(doubletId);
        auto const* __restrict__ end = cellNeighborsHisto->end(doubletId);
        auto const nInBin = end - bin;

        for (auto idx = 0u; idx < nInBin; idx++) {
          // FIXME implement alpaka::ldg and use it here? or is it const* __restrict__ enough?
          unsigned int otherCell = bin[idx];
        // for (unsigned int otherCell : outerNeighbors()) {
          printf("doubletId: %ld -> %d\n",doubletId,otherCell);
          if (cells[otherCell].isKilled())
            continue;  // killed by earlyFishbone
          last = false;
          cells[otherCell].template find_ntuplets<DEPTH - 1>(
              acc, hh, cc, cells, cellTracks, foundNtuplets, cellNeighborsHisto, cellTracksHisto, apc, quality, tmpNtuplet, minHitsPerNtuplet);
        }
        if (last) {  // if long enough save...
          if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
#ifdef ONLY_TRIPLETS_IN_HOLE
            // triplets accepted only pointing to the hole
            auto startAt0 = cc[this->layerPairId()].graph()[0] == 0;
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
              hits[nh] = theOuterHitId;
              auto it = foundNtuplets.bulkFill(acc, apc, hits, nh + 1);
              printf("track n. %d nhits %d \n",it,nh+1);
              if (it >= 0) {  // if negative is overflow....
                for (auto c : tmpNtuplet)
                {
                  cells[c].addTrack(acc, it, cellTracks);
                  cellTracksHisto->count(acc,c);
                }
                quality[it] = bad;  // initialize to bad
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

    int16_t theLayerPairId_;
    uint8_t theInnerLayer_;
    uint8_t theOuterLayer_;
    uint16_t theStatus_;  // tbd

    float theInnerZ;
    float theInnerR;
    hindex_type theInnerHitId;
    hindex_type theOuterHitId;
    hindex_type theFishboneId;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CACell_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

// #define GPU_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS

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
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"

#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::caStructures;

  template <typename TrackerTraits>
  class CACell {
  public:
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(const HitsMultiView& hh,
                                             uint layerPairId,
                                             hindex_type innerHitId,
                                             hindex_type outerHitId) {
      // status bits start cleared (high byte of innerHitAndStatus_ left at 0)
      innerHitAndStatus_ = innerHitId & kHitIdMask;
      outerHitAndLayerPair_ = (outerHitId & kHitIdMask) | ((uint32_t(layerPairId) & kFieldMask) << kFieldShift);
      theFishboneId_ = invalidHitId;
    }

    using hindex_type = ::caStructures::hindex_type;
    using tindex_type = ::caStructures::tindex_type;

    static constexpr auto invalidHitId = std::numeric_limits<hindex_type>::max();

    using TmpTuple = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxLayersPerTrack>;
    using HitContainer = caStructures::SequentialContainer;
    using CellToCell = caStructures::GenericContainer;
    using CellToTracks = caStructures::GenericContainer;
    using CAPairSoAView = caStructures::CAPairSoAView;

    using Quality = ::pixelTrack::Quality;
    static constexpr auto bad = ::pixelTrack::Quality::bad;

    static constexpr float kUninitializeCurvature = std::numeric_limits<float>::max();

    enum class StatusBit : uint8_t { kUsed = 1, kInTrack = 2, kKilled = 1 << 7 };

    CACell() = default;

    constexpr unsigned int inner_hit_id() const { return innerHitAndStatus_ & kHitIdMask; }
    constexpr unsigned int outer_hit_id() const { return outerHitAndLayerPair_ & kHitIdMask; }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void kill() {
      // Non-atomic OR on the status byte (bits [24,32) of innerHitAndStatus_): the hit-id bits are written
      // once in init() and never touched again, and racing writers all set the same status bit.
      innerHitAndStatus_ |= (uint32_t(uint8_t(StatusBit::kKilled)) << kFieldShift);
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isKilled() const {
      return ((innerHitAndStatus_ >> kFieldShift) & uint8_t(StatusBit::kKilled)) != 0u;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint8_t layerPairId() const {
      return uint8_t((outerHitAndLayerPair_ >> kFieldShift) & kFieldMask);
    }
    // The CA-layer indices are not cached on the cell but looked up on demand from the layer-pair
    // graph, whose SoA holds one row per configured layer pair.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint8_t innerLayer(::reco::CAGraphSoAConstView cc) const {
      return static_cast<uint8_t>(cc[layerPairId()].layerPair()[0]);
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint8_t outerLayer(::reco::CAGraphSoAConstView cc) const {
      return static_cast<uint8_t>(cc[layerPairId()].layerPair()[1]);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool unused() const {
      return 0 == (uint8_t(StatusBit::kUsed) & uint8_t((innerHitAndStatus_ >> kFieldShift) & kFieldMask));
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setStatusBits(StatusBit mask) {
      // Non-atomic OR on the status byte; see kill() for the race rationale.
      innerHitAndStatus_ |= (uint32_t(uint8_t(mask)) << kFieldShift);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_x(const HitsMultiView& hh) const { return hh[inner_hit_id()].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_x(const HitsMultiView& hh) const { return hh[outer_hit_id()].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_y(const HitsMultiView& hh) const { return hh[inner_hit_id()].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_y(const HitsMultiView& hh) const { return hh[outer_hit_id()].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_z(const HitsMultiView& hh) const { return hh[inner_hit_id()].zGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_z(const HitsMultiView& hh) const { return hh[outer_hit_id()].zGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_r(const HitsMultiView& hh) const { return hh[inner_hit_id()].rGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_r(const HitsMultiView& hh) const { return hh[outer_hit_id()].rGlobal(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_iphi(const HitsMultiView& hh) const { return hh[inner_hit_id()].iphi(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto outer_iphi(const HitsMultiView& hh) const { return hh[outer_hit_id()].iphi(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_dPhiDr(const HitsMultiView& hh) const {
      return hh[inner_hit_id()].dPhiDr();
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_isStub(const HitsMultiView& hh) const {
      return isStub(hh, inner_hit_id());
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto outer_isStub(const HitsMultiView& hh) const {
      return isStub(hh, outer_hit_id());
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_detIndex(const HitsMultiView& hh) const {
      return hh[inner_hit_id()].detectorIndex();
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_detIndex(const HitsMultiView& hh) const {
      return hh[outer_hit_id()].detectorIndex();
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto fishboneId() const { return theFishboneId_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hasFishbone() const { return theFishboneId_ != invalidHitId; }

    ALPAKA_FN_ACC void print_cell() const {
      printf("printing cell: on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
             layerPairId(),
             inner_hit_id(),
             outer_hit_id());
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setFishbone(Acc2D const& acc, hindex_type id, float z, const HitsMultiView& hh) {
      // make it deterministic: use the farther apart (in z), breaking exact ties by the hit id so
      // that concurrent updates converge to the same winner independently of their order
      const float zi = inner_z(hh);
      auto old = theFishboneId_;
      while (old != alpaka::atomicCas(acc,
                                      &theFishboneId_,
                                      old,
                                      (invalidHitId == old || std::abs(z - zi) > std::abs(hh[old].zGlobal() - zi) ||
                                       (std::abs(z - zi) == std::abs(hh[old].zGlobal() - zi) && id < old))
                                          ? id
                                          : old,
                                      alpaka::hierarchy::Blocks{}))
        old = theFishboneId_;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto quadrupletCut(const float innerCurvature,
                                                      float& outerCurvature,
                                                      const ::reco::CANtupletCutsSoAConstView& ntupletCuts,
                                                      const ::reco::CAGraphSoAConstView& cc,
                                                      const HitsMultiView& hh,
                                                      const float x1,
                                                      const float y1) const {
      // calculate curvature for the XY plane cuts
      float x2 = inner_x(hh);
      float y2 = inner_y(hh);
      float x3 = outer_x(hh);
      float y3 = outer_y(hh);
      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);
      outerCurvature = eq.curvature();

      if (innerCurvature == kUninitializeCurvature)
        return false;

      const auto ol = outerLayer(cc);
      auto maxDCurv = ntupletCuts[ol].maxDCurv();
      auto dCurv0 = ntupletCuts[ol].floorDCurv();

#ifdef CA_DEBUG
      printf(
          "quadCut: layer=%d, dCurv=%f, curv0=%f, Co=%f, Ci=%f", ol, maxDCurv, dCurv0, outerCurvature, innerCurvature);
#endif
      // linear cut
      return std::abs(outerCurvature - innerCurvature) >
             maxDCurv * (std::abs(innerCurvature + outerCurvature)) + dCurv0;
    }

    // trying to free the track building process from hardcoded layers, leaving
    // the visit of the graph based on the neighborhood connections between cells.

    template <int DEPTH>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void find_ntuplets(Acc1D const& acc,
                                                      const HitsMultiView& hh,
                                                      const ::reco::CANtupletCutsSoAConstView& ntupletCuts,
                                                      const ::reco::CAGraphSoAConstView& cc,
                                                      CACell* __restrict__ cells,
                                                      HitContainer& foundNtuplets,
                                                      CellToCell const* __restrict__ cellNeighborsHisto,
                                                      CellToTracks* cellTracksHisto,
                                                      uint32_t* nCellTracks,
                                                      CAPairSoAView ct,
                                                      cms::alpakatools::AtomicPairCounter& apc,
                                                      Quality* __restrict__ quality,
                                                      int8_t* __restrict__ nLayers,
                                                      float* __restrict__ pt,
                                                      TmpTuple& tmpNtuplet,
                                                      hindex_type* __restrict__ hits,
                                                      const unsigned int minHitsPerNtuplet,
                                                      const float preCurvature = kUninitializeCurvature) const {
      // the building process for a track ends if:
      // it has no right neighbor
      // it has no compatible neighbor
      // the ntuplets is then saved if the number of hits it contains is greater
      // than a threshold
      if constexpr (DEPTH <= 0) {
        printf("ERROR: CACell::find_ntuplets reached full depth!\n");
        ALPAKA_ASSERT_ACC(false);
      } else {
        auto doubletId = this - cells;
        tmpNtuplet.push_back_unsafe(doubletId);  // if we move this to be safe we could parallelize further below?
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() <= int(TrackerTraits::maxLayersPerTrack - 1));

        // Single bin per cell. Each stored entry encodes the neighbor cell index
        // in its low 31 bits and the layer-skipping flag in bit 31. Walk the
        // bin in two passes so that non-skipping neighbors are tried first and
        // skipping ones are only explored when no good non-skipping neighbor
        // was found at triplet-or-more depth.
        auto const* __restrict__ neighborCells = cellNeighborsHisto->begin(doubletId);
        auto const nNeighbors = cellNeighborsHisto->size(doubletId);

        bool tripletOrMore = ((unsigned int)(tmpNtuplet.size()) > 1);
        bool foundNeighbor = false;

        // The two passes are identical but for the check on bit 31 (layer skipping):
        // pass==0 keeps entries with bit 31 clear, pass==1 keeps entries with
        // bit 31 set. Bail out of pass 1 if a non-skipping neighbor already
        // produced a triplet-or-more.
        for (int pass = 0; pass < 2; ++pass) {
          if (pass == 1 && foundNeighbor && tripletOrMore)
            break;
          const uint32_t wantSkip = (pass == 0) ? 0u : caStructures::kSkipsLayerFlag;
          for (auto idx = 0u; idx < nNeighbors; idx++) {
            auto encoded = neighborCells[idx];
            if ((encoded & caStructures::kSkipsLayerFlag) != wantSkip)
              continue;
            auto otherCell = encoded & caStructures::kCellIndexMask;
            if (cells[otherCell].isKilled())
              continue;

            float thisCurvature{kUninitializeCurvature};
            if (cells[otherCell].quadrupletCut(
                    preCurvature, thisCurvature, ntupletCuts, cc, hh, inner_x(hh), inner_y(hh)))
              continue;
#ifdef CA_DEBUG
            printf("Doublet no. %d %d doubletId: %ld -> %d (isKilled %d) (%d,%d) -> (%d,%d) %d %d\n",
                   tmpNtuplet.size(),
                   idx,
                   doubletId,
                   otherCell,
                   cells[otherCell].isKilled(),
                   this->inner_hit_id(),
                   this->outer_hit_id(),
                   cells[otherCell].inner_hit_id(),
                   cells[otherCell].outer_hit_id(),
                   idx,
                   nNeighbors);
#endif

            foundNeighbor = true;
            cells[otherCell].template find_ntuplets<DEPTH - 1>(acc,
                                                               hh,
                                                               ntupletCuts,
                                                               cc,
                                                               cells,
                                                               foundNtuplets,
                                                               cellNeighborsHisto,
                                                               cellTracksHisto,
                                                               nCellTracks,
                                                               ct,
                                                               apc,
                                                               quality,
                                                               nLayers,
                                                               pt,
                                                               tmpNtuplet,
                                                               hits,
                                                               minHitsPerNtuplet,
                                                               thisCurvature);
          }
        }

        // if no more doublets were found, save N-tuplet if long enough
        if (!foundNeighbor) {
          const uint8_t nl = tmpNtuplet.size() + 1;  // numLayers in tuplet
          // if long enough save...
          if (nl >= minHitsPerNtuplet) {
            {
              // `hits` is owned by the caller, one buffer per thread sized maxHitsOnTrack to cover fishbone
              // hits: declaring it here would put a copy at every inlined recursion depth on the stack, and a
              // thread only ever saves one ntuplet at a time.
              uint32_t nh = 0U;
              uint32_t nfb = 0U;
              for (auto c : tmpNtuplet) {
                hits[nh++] = cells[c].inner_hit_id();
                if (nfb < TrackerTraits::maxFishboneHitsPerTrack && cells[c].hasFishbone()) {
                  ++nfb;
                  hits[nh++] = cells[c].theFishboneId_;  // Fishbone hit is always outer than inner hit
                }
              }
              ALPAKA_ASSERT_ACC(nh < TrackerTraits::maxHitsOnTrack);
              hits[nh] = outer_hit_id();
              auto it = foundNtuplets.bulkFill(acc, apc, hits, nh + 1);
#ifdef CA_DEBUG
              printf("track n. %d nhits %d with cells: ", it, nh + 1);
#endif
              if (it != cms::alpakatools::kOverflow) {
                for (auto c : tmpNtuplet) {
#ifdef CA_DEBUG
                  printf("%d - ", c);
#endif
                  auto t_ind = alpaka::atomicAdd(acc, nCellTracks, 1u, alpaka::hierarchy::Blocks{});

                  if (t_ind >= uint32_t(ct.metadata().size())) {
#ifdef CA_WARNINGS
                    printf("Warning!!!! Too many cell->tracks associations (limit = %d)!\n", ct.metadata().size());
#endif
                    alpaka::atomicSub(acc, nCellTracks, 1u, alpaka::hierarchy::Blocks{});
                    break;
                  }
                  // Key-range guard: one bin per cell, so this holds by construction and only a sizing
                  // mismatch can fail it, in which case the association is dropped rather than written
                  // outside off[].
                  if (c < cellTracksHisto->nOnes())
                    cellTracksHisto->count(acc, c);

                  ct[t_ind].inner() = c;   // cell
                  ct[t_ind].outer() = it;  // track
                }
#ifdef CA_DEBUG
                printf("\n");
#endif

                nLayers[it] = int8_t(nl);  // set number of layers in the TrackSoA (if not done here,
                                           // one would need to recalculate it from the hits later)
                quality[it] = bad;
                pt[it] = preCurvature;  // fill the curvature as an early (pre-fit)
                                        // reference for pt comparisons in duplicate removers
              }
#ifdef CA_WARNINGS
              else {
                printf("Warning!!!! Too many tuples (nOnes = %d)!\n", static_cast<int>(foundNtuplets.nOnes()));
              }
#endif
            }
          }
        }
        tmpNtuplet.pop_back();
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() < int(TrackerTraits::maxHitsOnTrack - 1));
      }
    }

  private:
    // Packed 12-byte layout:
    //   innerHitAndStatus_ : bits [0,24) innerHitId | bits [24,32) status
    //   outerHitAndLayerPair_ : bits [0,24) outerHitId | bits [24,32) layerPairId
    //   theFishboneId_ : standalone word, so setFishbone's atomicCas targets a dedicated address
    // Hit indices fit in 24 bits (16.7M >> ~3e5 hits/event), the status in 8 (kKilled = 1<<7).
    static constexpr uint32_t kHitIdMask = 0x00FFFFFFu;
    static constexpr uint32_t kFieldShift = 24u;
    static constexpr uint32_t kFieldMask = 0xFFu;

    uint32_t innerHitAndStatus_;
    uint32_t outerHitAndLayerPair_;
    hindex_type theFishboneId_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

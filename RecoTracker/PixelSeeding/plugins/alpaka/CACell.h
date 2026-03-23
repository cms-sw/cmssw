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
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void init(const HitsConstView& hh,
                                             int layerPairId,
                                             uint8_t theInnerLayer,
                                             uint8_t theOuterLayer,
                                             hindex_type innerHitId,
                                             hindex_type outerHitId) {
      theInnerHitId_ = innerHitId;
      theOuterHitId_ = outerHitId;
      theLayerPairId_ = layerPairId;
      theInnerLayer_ = theInnerLayer;
      theOuterLayer_ = theOuterLayer;
      theStatus_ = 0;
      theFishboneId_ = invalidHitId;

      // optimization that depends on access pattern
      theInnerZ_ = hh[innerHitId].zGlobal();
      theInnerR_ = hh[innerHitId].rGlobal();
    }

    using hindex_type = ::caStructures::hindex_type;
    using tindex_type = ::caStructures::tindex_type;

    static constexpr auto invalidHitId = std::numeric_limits<hindex_type>::max();

    using TmpTuple = cms::alpakatools::VecArray<uint32_t, TrackerTraits::maxDepth>;
    using HitContainer = caStructures::SequentialContainer;
    using CellToCell = caStructures::GenericContainer;
    using CellToTracks = caStructures::GenericContainer;
    using CAPairSoAView = caStructures::CAPairSoAView;

    using Quality = ::pixelTrack::Quality;
    static constexpr auto bad = ::pixelTrack::Quality::bad;

    enum class StatusBit : uint16_t { kUsed = 1, kInTrack = 2, kKilled = 1 << 15 };

    CACell() = default;

    constexpr unsigned int inner_hit_id() const { return theInnerHitId_; }
    constexpr unsigned int outer_hit_id() const { return theOuterHitId_; }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void kill() { theStatus_ |= uint16_t(StatusBit::kKilled); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isKilled() const { return theStatus_ & uint16_t(StatusBit::kKilled); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t layerPairId() const { return theLayerPairId_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t innerLayer() const { return theInnerLayer_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int16_t outerLayer() const { return theOuterLayer_; }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool unused() const { return 0 == (uint16_t(StatusBit::kUsed) & theStatus_); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setStatusBits(StatusBit mask) { theStatus_ |= uint16_t(mask); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_x(const HitsConstView& hh) const { return hh[theInnerHitId_].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_x(const HitsConstView& hh) const { return hh[theOuterHitId_].xGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_y(const HitsConstView& hh) const { return hh[theInnerHitId_].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_y(const HitsConstView& hh) const { return hh[theOuterHitId_].yGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_z(const HitsConstView& hh) const { return theInnerZ_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_z(const HitsConstView& hh) const { return hh[theOuterHitId_].zGlobal(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_r(const HitsConstView& hh) const { return theInnerR_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_r(const HitsConstView& hh) const { return hh[theOuterHitId_].rGlobal(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_iphi(const HitsConstView& hh) const { return hh[theInnerHitId_].iphi(); }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto outer_iphi(const HitsConstView& hh) const { return hh[theOuterHitId_].iphi(); }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float inner_detIndex(const HitsConstView& hh) const {
      return hh[theInnerHitId_].detectorIndex();
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float outer_detIndex(const HitsConstView& hh) const {
      return hh[theOuterHitId_].detectorIndex();
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto fishboneId() const { return theFishboneId_; }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool hasFishbone() const { return theFishboneId_ != invalidHitId; }

    ALPAKA_FN_ACC void print_cell() const {
      printf("printing cell: on layerPair: %d, innerLayer: %d, outerLayer: %d, innerHitId: %d, outerHitId: %d \n",
             theLayerPairId_,
             theInnerLayer_,
             theOuterLayer_,
             theInnerHitId_,
             theOuterHitId_);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void setFishbone(Acc2D const& acc, hindex_type id, float z, const HitsConstView& hh) {
      // make it deterministic: use the farther apart (in z)
      auto old = theFishboneId_;
      while (
          old !=
          alpaka::atomicCas(
              acc,
              &theFishboneId_,
              old,
              (invalidHitId == old || std::abs(z - theInnerZ_) > std::abs(hh[old].zGlobal() - theInnerZ_)) ? id : old,
              alpaka::hierarchy::Blocks{}))
        old = theFishboneId_;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool areAlignedRZ(
        float r1, float z1, float ri, float zi, float ro, float zo, const float ptmin, const float thetaCut) {
      float radius_diff = std::abs(r1 - ro);
      float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

      float pMin = ptmin * std::sqrt(distance_13_squared);  // this needs to be divided by
                                                            // radius_diff later

      float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
      return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool dcaCut(const HitsConstView& hh,
                                               CACell const& otherCell,
                                               const float region_origin_radius_plus_tolerance,
                                               const float maxCurv) const {
      auto x1 = otherCell.inner_x(hh);
      auto y1 = otherCell.inner_y(hh);

      auto x2 = inner_x(hh);
      auto y2 = inner_y(hh);

      auto x3 = outer_x(hh);
      auto y3 = outer_y(hh);

      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);

      if (std::abs(eq.curvature()) > maxCurv)
        return false;

      return std::abs(eq.dca0()) < region_origin_radius_plus_tolerance * std::abs(eq.curvature());
    }

    // trying to free the track building process from hardcoded layers, leaving
    // the visit of the graph based on the neighborhood connections between cells.

    template <int DEPTH>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void find_ntuplets(Acc1D const& acc,
                                                      const ::reco::CAGraphSoAConstView& cc,
                                                      CACell* __restrict__ cells,
                                                      HitContainer& foundNtuplets,
                                                      CellToCell const* __restrict__ cellNeighborsHisto,
                                                      CellToTracks* cellTracksHisto,
                                                      uint32_t* nCellTracks,
                                                      CAPairSoAView ct,
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
        printf("ERROR: CACell::find_ntuplets reached full depth!\n");
        ALPAKA_ASSERT_ACC(false);
      } else {
        auto doubletId = this - cells;
        tmpNtuplet.push_back_unsafe(doubletId);  // if we move this to be safe we could parallelize further below?
        ALPAKA_ASSERT_ACC(tmpNtuplet.size() <= int(TrackerTraits::maxHitsOnTrack - 3));

        bool last = true;
        auto const* __restrict__ bin = cellNeighborsHisto->begin(doubletId);
        auto nInBin = cellNeighborsHisto->size(doubletId);

        for (auto idx = 0u; idx < nInBin; idx++) {
          // FIXME implement alpaka::ldg and use it here? or is it const* __restrict__ enough?
          unsigned int otherCell = bin[idx];
          if (cells[otherCell].isKilled())
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
                 nInBin);
#endif

          last = false;
          cells[otherCell].template find_ntuplets<DEPTH - 1>(acc,
                                                             cc,
                                                             cells,
                                                             foundNtuplets,
                                                             cellNeighborsHisto,
                                                             cellTracksHisto,
                                                             nCellTracks,
                                                             ct,
                                                             apc,
                                                             quality,
                                                             tmpNtuplet,
                                                             minHitsPerNtuplet);
        }
        if (last) {  // if long enough save...
          if ((unsigned int)(tmpNtuplet.size()) >= minHitsPerNtuplet - 1) {
            {
              hindex_type hits[TrackerTraits::maxDepth + 2];
              auto nh = 0U;
              constexpr int maxFB = 2;  // for the time being let's limit this
              int nfb = 0;
              for (auto c : tmpNtuplet) {
                hits[nh++] = cells[c].theInnerHitId_;
                if (nfb < maxFB && cells[c].hasFishbone()) {
                  ++nfb;
                  hits[nh++] = cells[c].theFishboneId_;  // Fishbone hit is always outer than inner hit
                }
              }
              ALPAKA_ASSERT_ACC(nh < TrackerTraits::maxHitsOnTrack);
              hits[nh] = theOuterHitId_;
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
                  cellTracksHisto->count(acc, c);

                  ct[t_ind].inner() = c;   //cell
                  ct[t_ind].outer() = it;  //track
                }
#ifdef CA_DEBUG
                printf("\n");
#endif
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
    int16_t theLayerPairId_;
    uint8_t theInnerLayer_;
    uint8_t theOuterLayer_;
    uint16_t theStatus_;  // tbd

    float theInnerZ_;
    float theInnerR_;
    hindex_type theInnerHitId_;
    hindex_type theOuterHitId_;
    hindex_type theFishboneId_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

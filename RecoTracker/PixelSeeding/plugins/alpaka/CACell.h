#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CACell_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CACell_h

// #define GPU_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS

#include <cmath>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
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
    // The hit view this topology's CA reads (HitsMultiView, or the CAHitsView facade for
    // Phase2OTVectorHits). Declared first: every signature below names it.
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

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

    // Bend of the inner hit. Vector-hit-only: the caller must have established isVectorHit(hh, inner_hit_id()).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_dPhiDr(const HitsMultiView& hh) const {
      return hh.vectorHit(inner_hit_id()).dPhiDr();
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto inner_isVectorHit(const HitsMultiView& hh) const {
      return isVectorHit(hh, inner_hit_id());
    }
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto outer_isVectorHit(const HitsMultiView& hh) const {
      return isVectorHit(hh, outer_hit_id());
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
                                      alpaka::hierarchy::Blocks{})) {
        old = theFishboneId_;
      }
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

    // ---- Search of the ntuplets ----
    // The ntuplets are built by a visit of the cell graph that follows the neighbour connections between cells, instead
    // of hardcoded layers. From each cell the visit tries its compatible neighbours, the non layer-skipping ones first;
    // the layer-skipping ones are tried only if none of those was found at triplet-or-more depth. A cell without any
    // compatible neighbour ends an ntuplet, which is saved if it has enough hits.

    // Save the ntuplet made of the cells path[0..len), whose last cell was reached with the pre-fit curvature
    // curvature, if it has at least minHitsPerNtuplet layers. path is indexable (a pointer, or GlobalFrames::Path).
    // hits is a per-thread buffer of maxHitsOnTrack entries, owned by the caller.
    template <typename Path>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void save_ntuplet(Acc1D const& acc,
                                                            const HitsMultiView& hh,
                                                            CACell* __restrict__ cells,
                                                            HitContainer& foundNtuplets,
                                                            CellToTracks* cellTracksHisto,
                                                            uint32_t* nCellTracks,
                                                            CAPairSoAView ct,
                                                            cms::alpakatools::AtomicPairCounter& apc,
                                                            Quality* __restrict__ quality,
                                                            int8_t* __restrict__ nLayers,
                                                            float* __restrict__ pt,
                                                            Path const& path,
                                                            int len,
                                                            float curvature,
                                                            hindex_type* __restrict__ hits,
                                                            const unsigned int minHitsPerNtuplet) {
      const uint8_t nl = len + 1;  // numLayers in tuplet
      if (nl < minHitsPerNtuplet)
        return;
      uint32_t nh = 0U;
      uint32_t nfb = 0U;
      for (int k = 0; k < len; ++k) {
        auto c = path[k];
        hits[nh++] = cells[c].inner_hit_id();
        if (nfb < TrackerTraits::maxFishboneHitsPerTrack && cells[c].hasFishbone()) {
          ++nfb;
          hits[nh++] = cells[c].theFishboneId_;  // Fishbone hit is always outer than inner hit
        }
      }
      ALPAKA_ASSERT_ACC(nh < TrackerTraits::maxHitsOnTrack);
      hits[nh] = cells[path[len - 1]].outer_hit_id();
      auto it = foundNtuplets.bulkFill(acc, apc, hits, nh + 1);
      if (it != cms::alpakatools::kOverflow) {
        for (int k = 0; k < len; ++k) {
          auto c = path[k];
          auto t_ind = alpaka::atomicAdd(acc, nCellTracks, 1u, alpaka::hierarchy::Blocks{});
          if (t_ind >= uint32_t(ct.metadata().size())) {
#ifdef CA_WARNINGS
            printf("Warning!!!! Too many cell->tracks associations (limit = %d)!\n", ct.metadata().size());
#endif
            alpaka::atomicSub(acc, nCellTracks, 1u, alpaka::hierarchy::Blocks{});
            break;
          }
          // Key-range guard: one bin per cell, so this holds by construction and only a sizing mismatch can fail it,
          // in which case the association is dropped rather than written outside off[].
          if (c < cellTracksHisto->nOnes()) {
            cellTracksHisto->count(acc, c);
          }
          ct[t_ind].inner() = c;   // cell
          ct[t_ind].outer() = it;  // track
        }
        nLayers[it] = int8_t(nl);  // number of layers, so that it does not need to be recomputed from the hits
        quality[it] = bad;
        pt[it] = curvature;  // pre-fit curvature: an early reference for the pt comparisons of the duplicate removers
      }
#ifdef CA_WARNINGS
      else {
        printf("Warning!!!! Too many tuples (nOnes = %d)!\n", static_cast<int>(foundNtuplets.nOnes()));
      }
#endif
    }

    // Frame stores of the depth-first search find_ntuplets: on the stack of the thread (StackFrames), or in global
    // memory as structure of arrays with one slot per search (GlobalFrames, see GlobalFrameBuffers). path() gives the
    // cells of the frames, for save_ntuplet.
    template <int MAXDEPTH>
    struct StackFrames {
      uint32_t cell_[MAXDEPTH];
      uint32_t next_[MAXDEPTH];
      float curvature_[MAXDEPTH];
      uint8_t pass_[MAXDEPTH];
      bool found_[MAXDEPTH];
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t& cell(int l) { return cell_[l]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t& next(int l) { return next_[l]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE float& curvature(int l) { return curvature_[l]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint8_t& pass(int l) { return pass_[l]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool& found(int l) { return found_[l]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t const* path() const { return cell_; }
    };

    // the arrays of the global frame store: entry (l, slot) at l * stride + slot, and maxHitsOnTrack hits per slot
    struct GlobalFrameBuffers {
      uint32_t* cell;
      uint32_t* next;
      float* curvature;
      uint8_t* pass;
      bool* found;
      hindex_type* hits;
      uint32_t stride;  // number of slots
    };

    struct GlobalFrames {
      struct Path {
        uint32_t const* cell;
        uint32_t stride;
        uint32_t slot;
        ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t operator[](int l) const { return cell[l * stride + slot]; }
      };
      GlobalFrameBuffers b;
      uint32_t slot;
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t& cell(int l) { return b.cell[l * b.stride + slot]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t& next(int l) { return b.next[l * b.stride + slot]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE float& curvature(int l) { return b.curvature[l * b.stride + slot]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint8_t& pass(int l) { return b.pass[l * b.stride + slot]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE bool& found(int l) { return b.found[l * b.stride + slot]; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE Path path() const { return Path{b.cell, b.stride, slot}; }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE hindex_type* hits() const { return b.hits + slot * TrackerTraits::maxHitsOnTrack; }
    };

    // Depth-first search of the subtree of the cell startCell, reached through the cells prefix[0..prefixLen) with the
    // pre-fit curvature startCurvature, saving the ntuplets it finds. A search from a starting cell has an empty
    // prefix and kUninitializeCurvature; the neighbour loop of startCell starts at (startPass, startNext, startFound),
    // from the beginning or where the level-by-level search stopped. The visit uses an explicit stack of frames (fr,
    // StackFrames or GlobalFrames), one per depth, instead of a recursion that the compiler would inline once per
    // depth. Used by Kernel_find_ntuplets (back-ends with a single thread per block) and by
    // Kernel_find_ntuplets_overflow, when the node storage of the level-by-level search is full.
    template <int MAXDEPTH, typename Frames>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void find_ntuplets(Acc1D const& acc,
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
                                                             Frames& fr,
                                                             hindex_type* __restrict__ hits,
                                                             const unsigned int minHitsPerNtuplet,
                                                             uint32_t const* prefix,
                                                             int prefixLen,
                                                             uint32_t startCell,
                                                             float startCurvature,
                                                             uint8_t startPass = 0,
                                                             uint32_t startNext = 0,
                                                             bool startFound = false) {
      static_assert(MAXDEPTH > 0);
      // one frame per depth: its cell (the path of the search is fr.cell(0..level]), where the neighbour loop of the
      // cell is (next: next neighbour index to try; pass: 0 for the non layer-skipping neighbours, 1 for the
      // layer-skipping ones, 2 when done), whether a compatible neighbour was found (and visited), and the pre-fit
      // curvature with which the cell was reached

      // the cells of the prefix are done: when the search returns to them it stops, without saving them
      for (int l = 0; l < prefixLen; ++l) {
        fr.cell(l) = prefix[l];
        fr.next(l) = 0;
        fr.pass(l) = 2;
        fr.found(l) = true;
        fr.curvature(l) = kUninitializeCurvature;
      }
      int level = prefixLen;
      fr.cell(level) = startCell;
      fr.next(level) = startNext;
      fr.pass(level) = startPass;
      fr.found(level) = startFound;
      fr.curvature(level) = startCurvature;

      while (level >= prefixLen) {
        uint32_t const doubletId = fr.cell(level);
        CACell const& cell = cells[doubletId];
        auto const* __restrict__ neighborCells = cellNeighborsHisto->begin(doubletId);
        auto const nNeighbors = cellNeighborsHisto->size(doubletId);
        bool const tripletOrMore = (level > 0);
        // at the last depth the ntuplet has maxLayersPerTrack layers: it is saved as it is, not extended
        bool const lastDepth = (level == MAXDEPTH - 1);

        // look for the next compatible neighbour, the non layer-skipping ones first
        int child = -1;
        float childCurvature = kUninitializeCurvature;
        while (not lastDepth and fr.pass(level) < 2) {
          // the layer-skipping pass is skipped if, when it would start, a non layer-skipping neighbour has already
          // been found at triplet-or-more depth
          if (fr.pass(level) == 1 && fr.next(level) == 0 && fr.found(level) && tripletOrMore) {
            fr.pass(level) = 2;
            break;
          }
          const uint32_t wantSkip = (fr.pass(level) == 0) ? 0u : caStructures::kSkipsLayerFlag;
          while (fr.next(level) < nNeighbors) {
            auto encoded = neighborCells[fr.next(level)++];
            if ((encoded & caStructures::kSkipsLayerFlag) != wantSkip)
              continue;
            auto otherCell = encoded & caStructures::kCellIndexMask;
            if (cells[otherCell].isKilled())
              continue;
            float thisCurvature{kUninitializeCurvature};
            if (cells[otherCell].quadrupletCut(
                    fr.curvature(level), thisCurvature, ntupletCuts, cc, hh, cell.inner_x(hh), cell.inner_y(hh))) {
              continue;
            }
            child = otherCell;
            childCurvature = thisCurvature;
            break;
          }
          if (child >= 0)
            break;
          // this pass is over, start the next one from the first neighbour
          ++fr.pass(level);
          fr.next(level) = 0;
        }

        if (child >= 0) {
          // go down to the neighbour
          fr.found(level) = true;
          ++level;
          ALPAKA_ASSERT_ACC(level < MAXDEPTH);
          fr.cell(level) = child;
          fr.next(level) = 0;
          fr.pass(level) = 0;
          fr.found(level) = false;
          fr.curvature(level) = childCurvature;
          continue;
        }

        // all neighbours of this cell have been visited: if none was found, save the ntuplet if long enough
        if (!fr.found(level)) {
          save_ntuplet(acc,
                       hh,
                       cells,
                       foundNtuplets,
                       cellTracksHisto,
                       nCellTracks,
                       ct,
                       apc,
                       quality,
                       nLayers,
                       pt,
                       fr.path(),
                       level + 1,
                       fr.curvature(level),
                       hits,
                       minHitsPerNtuplet);
        }
        // go back up to the parent cell
        --level;
      }
    }

    // ---- Level-by-level search (GPU backends) ----
    // The same visit as find_ntuplets, one depth at a time: each node of the search tree (a cell reached through a path
    // of cells, with the pre-fit curvature passed to it) is expanded by one thread, which applies the neighbour loop of
    // one frame of the depth-first search and either records the compatible neighbours as nodes of the next depth or,
    // if there is none, saves the ntuplet.

    // Where expand_cell stopped because a neighbour could not be stored: the search of the cell resumes at neighbour
    // next of pass pass (that neighbour included), with found = whether a neighbour was stored before it.
    struct ExpandStop {
      bool stopped = false;
      uint8_t pass = 0;
      uint32_t next = 0;
      bool found = false;
    };

    // Calls emit(neighbour, curvature) for each compatible neighbour of the cell doubletId at depth level, reached
    // with the pre-fit curvature curvature, in the order and with the cuts of find_ntuplets, until emit returns false
    // (the neighbour could not be stored): then it fills stop and returns. Returns whether any compatible neighbour
    // was found. At the last depth the cell is not extended.
    template <int MAXDEPTH, typename F>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool expand_cell(const HitsMultiView& hh,
                                                           const ::reco::CANtupletCutsSoAConstView& ntupletCuts,
                                                           const ::reco::CAGraphSoAConstView& cc,
                                                           CACell* __restrict__ cells,
                                                           CellToCell const* __restrict__ cellNeighborsHisto,
                                                           uint32_t doubletId,
                                                           int level,
                                                           float curvature,
                                                           F&& emit,
                                                           ExpandStop& stop) {
      if (level == MAXDEPTH - 1)
        return false;
      CACell const& cell = cells[doubletId];
      auto const* __restrict__ neighborCells = cellNeighborsHisto->begin(doubletId);
      auto const nNeighbors = cellNeighborsHisto->size(doubletId);
      bool found = false;
      for (int pass = 0; pass < 2; ++pass) {
        // the layer-skipping pass is skipped if a non layer-skipping neighbour has been found at triplet-or-more depth
        if (pass == 1 && found && level > 0)
          break;
        const uint32_t wantSkip = (pass == 0) ? 0u : caStructures::kSkipsLayerFlag;
        for (uint32_t n = 0; n < nNeighbors; ++n) {
          auto encoded = neighborCells[n];
          if ((encoded & caStructures::kSkipsLayerFlag) != wantSkip)
            continue;
          auto otherCell = encoded & caStructures::kCellIndexMask;
          if (cells[otherCell].isKilled())
            continue;
          float thisCurvature{kUninitializeCurvature};
          if (cells[otherCell].quadrupletCut(
                  curvature, thisCurvature, ntupletCuts, cc, hh, cell.inner_x(hh), cell.inner_y(hh))) {
            continue;
          }
          if (not emit(otherCell, thisCurvature)) {
            stop = ExpandStop{true, static_cast<uint8_t>(pass), n, found};
            return true;
          }
          found = true;
        }
      }
      return found;
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

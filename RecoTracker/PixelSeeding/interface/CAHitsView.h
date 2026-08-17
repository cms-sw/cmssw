#ifndef RecoTracker_PixelSeeding_interface_CAHitsView_h
#define RecoTracker_PixelSeeding_interface_CAHitsView_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"

namespace caStructures {

  /**
   * @brief One global hit index over two SoA sources: pixel rechits and outer-tracker vector hits.
   *
   *     [0, nPixels)                 pixel rechits, in pixel-collection order
   *     [nPixels, nPixels + nVectorHits)  vector hits, in vector-hits-collection order (vectorHitIndex = i - nPixels)
   *
   * `hh[i]` returns the common element (only the columns both sources have: global position, r, iphi,
   * the global CA module index, the local errors). `hh.pixel(i)` and `hh.vectorHit(i)` return the
   * source-specific columns and assert the source of the index; there are no cross-source placeholder
   * values. Classify with isOTEntry(i) / isVectorHit(i) / hasBend(i), then read through the matching
   * accessor. Every access is range-checked with ALPAKA_ASSERT_ACC (free with NDEBUG).
   *
   * The two views are fixed members and every accessor selects a column base pointer with one compare
   * on the index; a run-time indexed array of views would spill to local memory in device code.
   * A third source means one more member, one more range in size()/moduleStart(), one more compare per
   * common accessor and one more typed accessor.
   */
  class CAHitsView {
  public:
    using PixelView = ::reco::TrackingRecHitConstView;
    using PixelModulesView = ::reco::HitModuleSoAConstView;
    using VectorHitsView = ::reco::VectorHitsConstView;
    using VectorHitModulesView = ::reco::VectorHitModuleConstView;
    using size_type = int32_t;

    static constexpr bool hasVectorHits = true;

    CAHitsView() = default;

    // nPixelModules is the number of pixel CA modules (the pixel moduleStart block has nPixelModules+1
    // entries); the outer-tracker CA modules follow, in the order of the vector hits' module block.
    CAHitsView(PixelView pixels,
               PixelModulesView pixelModules,
               VectorHitsView vectorHits,
               VectorHitModulesView vectorHitModules,
               uint32_t nPixels,
               uint32_t nVectorHits,
               uint32_t nPixelModules)
        : pixels_(pixels),
          pixelModules_(pixelModules),
          vectorHits_(vectorHits),
          vectorHitModules_(vectorHitModules),
          nPixels_(static_cast<int32_t>(nPixels)),
          nHits_(static_cast<int32_t>(nPixels + nVectorHits)),
          nPixelModules_(static_cast<int32_t>(nPixelModules)),
          nModules_(static_cast<int32_t>(nPixelModules) + static_cast<int32_t>(vectorHitModules.metadata().size()) -
                    1) {}

    // ---- sizes, ranges and predicates -----------------------------------------------------------
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE size_type size() const { return nHits_; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE size_type nPixels() const { return nPixels_; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE size_type nVectorHits() const { return nHits_ - nPixels_; }
    // First vector hit index (the number of pixel rechits).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t offsetVectorHits() const { return static_cast<uint32_t>(nPixels_); }

    // Which source a global index belongs to. Range-checked.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isOTEntry(int32_t i) const {
      checkRange(i);
      return i >= nPixels_;
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isPixel(int32_t i) const { return not isOTEntry(i); }
    // Outer-tracker entry with a valid bend error (>= 0). Entries with a negative bend error (published
    // single hits, degenerate vector hits) are treated as pixel-like hits.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isVectorHit(int32_t i) const {
      return isOTEntry(i) && vectorHits_.dPhiDrError()[i - nPixels_] >= 0.f;
    }
    // Does the entry carry a usable bend measurement (precision-only error > 0)?
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool hasBend(int32_t i) const {
      return isOTEntry(i) && vectorHits_.dPhiDrError()[i - nPixels_] > 0.f;
    }

    // The pixel view, for the offsetBPIX2 scalar (API-compatible with SoAConstMultiView::view(0)).
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PixelView const& view(int32_t) const { return pixels_; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PixelView const& pixels() const { return pixels_; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE VectorHitsView const& vectorHits() const { return vectorHits_; }

    // ---- module starts, as global hit indices --------------------------------------------------
    // m is a CA module index: pixel modules first, then the outer-tracker modules of the vector hits' block.
    // Valid for m in [0, nModules], the last entry being the total number of hits.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t moduleStart(int32_t m) const {
      ALPAKA_ASSERT_ACC(m >= 0 && m <= nModules_);
      return m < nPixelModules_ ? pixelModules_.moduleStart()[m]
                                : static_cast<uint32_t>(nPixels_) + vectorHitModules_.moduleStart()[m - nPixelModules_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE size_type nPixelModules() const { return nPixelModules_; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE size_type nModules() const { return nModules_; }

    // ---- common columns, by global index (both sources have them) ------------------------------
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float xGlobal(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.xGlobal()[i] : vectorHits_.xGlobal()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float yGlobal(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.yGlobal()[i] : vectorHits_.yGlobal()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float zGlobal(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.zGlobal()[i] : vectorHits_.zGlobal()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float rGlobal(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.rGlobal()[i] : vectorHits_.rGlobal()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int16_t iphi(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.iphi()[i] : vectorHits_.iphi()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint16_t detectorIndex(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.detectorIndex()[i] : vectorHits_.detectorIndex()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float xerrLocal(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.xerrLocal()[i] : vectorHits_.xerrLocal()[i - nPixels_];
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float yerrLocal(int32_t i) const {
      checkRange(i);
      return i < nPixels_ ? pixels_.yerrLocal()[i] : vectorHits_.yerrLocal()[i - nPixels_];
    }

    // ---- element proxies ------------------------------------------------------------------------
    // The common element: what hh[i] returns. Only columns every source has.
    class const_element {
    public:
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const_element(CAHitsView const& v, int32_t i) : v_(v), i_(i) {}
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float xGlobal() const { return v_.xGlobal(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float yGlobal() const { return v_.yGlobal(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float zGlobal() const { return v_.zGlobal(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float rGlobal() const { return v_.rGlobal(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int16_t iphi() const { return v_.iphi(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint16_t detectorIndex() const { return v_.detectorIndex(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float xerrLocal() const { return v_.xerrLocal(i_); }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float yerrLocal() const { return v_.yerrLocal(i_); }

    private:
      CAHitsView const& v_;
      int32_t i_;
    };
    using ConstElement = const_element;

    // Pixel-only columns. Constructed only through pixel(i), which asserts the source.
    class pixel_element {
    public:
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE pixel_element(PixelView const& p, int32_t i) : p_(p), i_(i) {}
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float xLocal() const { return p_.xLocal()[i_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float yLocal() const { return p_.yLocal()[i_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int16_t clusterSizeX() const { return p_.clusterSizeX()[i_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int16_t clusterSizeY() const { return p_.clusterSizeY()[i_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE SiPixelHitStatusAndCharge chargeAndStatus() const {
        return p_.chargeAndStatus()[i_];
      }

    private:
      PixelView const& p_;
      int32_t i_;  // pixel index == global index
    };

    // Vector-hit-only columns. Constructed only through vectorHit(i), which asserts the source.
    class vector_hit_element {
    public:
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE vector_hit_element(VectorHitsView const& s, int32_t j) : s_(s), j_(j) {}
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float dPhiDr() const { return s_.dPhiDr()[j_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float dPhiDrError() const { return s_.dPhiDrError()[j_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t lowerHitIdx() const { return s_.lowerHitIdx()[j_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t upperHitIdx() const { return s_.upperHitIdx()[j_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t posHitIdx() const { return s_.posHitIdx()[j_]; }
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint8_t flags() const { return s_.flags()[j_]; }
      // The vector hit's index in the vector hits collection.
      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int32_t vectorHitIndex() const { return j_; }

    private:
      VectorHitsView const& s_;
      int32_t j_;  // vector hit index == global index - nPixels
    };

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const_element operator[](int32_t i) const {
      checkRange(i);
      return const_element(*this, i);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE pixel_element pixel(int32_t i) const {
      ALPAKA_ASSERT_ACC(i >= 0 && i < nPixels_);
      return pixel_element(pixels_, i);
    }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE vector_hit_element vectorHit(int32_t i) const {
      ALPAKA_ASSERT_ACC(i >= nPixels_ && i < nHits_);
      return vector_hit_element(vectorHits_, i - nPixels_);
    }

  private:
    // Range contract of every accessor: 0 <= i < size(), checked with ALPAKA_ASSERT_ACC (active unless
    // NDEBUG). pixel(i) and vectorHit(i) assert the tighter source range; moduleStart(m) asserts 0 <= m <= nModules().
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void checkRange(int32_t i) const { ALPAKA_ASSERT_ACC(i >= 0 && i < nHits_); }

    PixelView pixels_;
    PixelModulesView pixelModules_;
    VectorHitsView vectorHits_;
    VectorHitModulesView vectorHitModules_;
    int32_t nPixels_ = 0;
    int32_t nHits_ = 0;
    int32_t nPixelModules_ = 0;
    int32_t nModules_ = 0;
  };

  // Free-function forms, so that generic code can say isVectorHit(hh, i) for any hit view type.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isVectorHit(CAHitsView const& hh, int32_t i) { return hh.isVectorHit(i); }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isOTEntry(CAHitsView const& hh, int32_t i) { return hh.isOTEntry(i); }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool hasBend(CAHitsView const& hh, int32_t i) { return hh.hasBend(i); }

}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_CAHitsView_h

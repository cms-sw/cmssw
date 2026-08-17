#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAFitHitSelection_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAFitHitSelection_h

#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"  // ::reco::isStub
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"          // ALPAKA_FN_ACC
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"                 // caOTHitTag::isOTId (tagged-OT-extra skip)

// Compile-time selection of which hits the BrokenLine fit uses (track finding stays IT+OT; only the fit and
// its multiplicity binning see the selected subset): All (default), ITonly (pixel hits only), OTonly (stub
// hits only). Edit kMode and recompile; tracks with < 3 selected hits are skipped.
namespace caFitHitSel {

  enum class Mode { All, ITonly, OTonly };

  constexpr Mode kMode = Mode::All;

  // Whether a hit (given its isStub flag) is used by the fit in the current mode.
  // hasStubs guards the case where the OT collections aren't present (offsetStubs < 0):
  // there, every hit is a pixel hit and we keep everything.
  constexpr inline bool useHit(bool isStub, bool hasStubs) {
    if (kMode == Mode::All || !hasStubs)
      return true;
    return (kMode == Mode::ITonly) ? !isStub : isStub;
  }

  // Overlap-module (same-layer) pixel hit de-duplication for the BL fit.
  constexpr float kDedupDsMin = 0.1f;  // cm; transverse merge radius (pixel-pixel only)
  constexpr float kDedupDsMin2 = kDedupDsMin * kDedupDsMin;

  // Single walk used by both the multiplicity counter and the fit's hit selection, so they cannot
  // disagree (a mismatch would trip the nSel in [nHitsL,nHitsH] assertions).
  //   k <  0 -> return the number of hits the fit will use (deduped, kMode-filtered).
  //   k >= 0 -> return the position j in [0,nhits) of the k-th kept hit.
  // excludeTaggedOT: a tagged OT-rechit extra id (bit30) is skipped rather than treated as an overflow
  // terminator. Such an id indexes the raw OT source, not the hit view, so it is neither part of
  // the count nor a terminator.
  template <typename TupleCont, typename HitsView>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t dedupWalk(TupleCont const* __restrict__ foundNtuplets,
                                                    uint32_t it,
                                                    HitsView hh,
                                                    bool hasStubs,
                                                    int k,
                                                    bool excludeTaggedOT = false) {
    auto const* hitId = foundNtuplets->begin(it);
    auto const nhits = foundNtuplets->size(it);
    auto const nTot = hh.size();
    uint32_t nkept = 0;
    int lastKeptJ = -1;
    for (uint32_t j = 0; j < uint32_t(nhits); ++j) {
      auto const h = hitId[j];
      if (h >= static_cast<unsigned int>(nTot)) {
        if (excludeTaggedOT && caOTHitTag::isOTId(h))
          continue;  // tagged OT extra -> not a merged-fit member; skip, keep walking
        break;       // content buffer overflow guard
      }
      // Stub-ness comes from reco::isStub(view, idx): stubs are the merged-collection hits at index
      // >= hh.offsetStubs().
      if (!useHit(::reco::isStub(hh, int32_t(h)), hasStubs))
        continue;
      // Merge only two consecutive pixel hits that are near-coincident in the transverse plane.
      if (hasStubs && lastKeptJ >= 0 && !::reco::isStub(hh, int32_t(h)) &&
          !::reco::isStub(hh, int32_t(hitId[lastKeptJ]))) {
        auto const hp = hitId[lastKeptJ];
        float const dx = float(hh[h].xGlobal()) - float(hh[hp].xGlobal());
        float const dy = float(hh[h].yGlobal()) - float(hh[hp].yGlobal());
        if (dx * dx + dy * dy < kDedupDsMin2)
          continue;  // overlap partner of the previous kept pixel -> merged (keep the earlier hit)
      }
      if (int(nkept) == k)
        return j;
      ++nkept;
      lastKeptJ = int(j);
    }
    return nkept;  // k < 0 (or out of range): the kept count
  }

}  // namespace caFitHitSel

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAFitHitSelection_h

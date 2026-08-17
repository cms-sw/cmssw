#ifndef RecoTracker_PixelSeeding_interface_OTHitTag_h
#define RecoTracker_PixelSeeding_interface_OTHitTag_h

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// OT-rechit tag for the CA fit-extension: a hit id with kOTHitTag set indexes the raw OT-rechit source
// instead of the merged TrackingRecHit SoA. Merged ids and OT indices are both < 2^30, so the tag never
// collides, and a tagged id sorts after a merged one, so the merged hit wins equal-chi2 ties.
namespace caOTHitTag {

  inline constexpr uint32_t kOTHitTag = 0x40000000u;
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isOTId(uint32_t id) { return (id & kOTHitTag) != 0u; }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t otIdx(uint32_t id) { return id & ~kOTHitTag; }

}  // namespace caOTHitTag

#endif  // RecoTracker_PixelSeeding_interface_OTHitTag_h

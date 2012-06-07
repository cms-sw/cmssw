#ifndef SimTracker_Common_SimHitInfoForLinks
#define SimTracker_Common_SimHitInfoForLinks

#include <vector>
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

// A stripped down version of PSimHit used to save memory.
// Contains only the information needed to be make DigiSimLinks.

  struct SimHitInfoForLinks {
    explicit SimHitInfoForLinks(PSimHit const* hitp) : eventId_(hitp->eventId()), trackIds_(1, hitp->trackId()) {
    }
    EncodedEventId eventId_;
    std::vector<unsigned int> trackIds_;
  };
#endif

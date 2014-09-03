#ifndef SimTracker_Common_SimHitInfoForLinks
#define SimTracker_Common_SimHitInfoForLinks

#include <vector>
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

// A stripped down version of PSimHit used to save memory.
// Contains only the information needed to be make DigiSimLinks.
// Include the simHit's index in the source collection, collection name suffix index.

  struct SimHitInfoForLinks {
    explicit SimHitInfoForLinks(PSimHit const* hitp, size_t hitindex, unsigned int tofbin) :
      eventId_(hitp->eventId()), trackIds_(1, hitp->trackId()), hitIndex_(hitindex), tofBin_(tofbin) {
    }
    EncodedEventId eventId_;
    std::vector<unsigned int> trackIds_;
    size_t hitIndex_;
    unsigned int tofBin_;
  };
#endif

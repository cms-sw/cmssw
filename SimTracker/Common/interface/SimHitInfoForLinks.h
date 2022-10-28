#ifndef SimTracker_Common_SimHitInfoForLinks
#define SimTracker_Common_SimHitInfoForLinks

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <vector>

// A stripped down version of PSimHit used to save memory.
// Contains only the information needed to be make DigiSimLinks.
// Include the simHit's index in the source collection, collection name suffix
// index.

class SimHitInfoForLinks {
public:
  explicit SimHitInfoForLinks(PSimHit const *hitp, size_t hitindex, unsigned int tofbin, size_t hitInd4CR, float amp)
      : eventId_(hitp->eventId()),
        trackIds_(1, hitp->trackId()),
        hitIndex_(hitindex),
        tofBin_(tofbin),
        hitInd4CR_(hitInd4CR),
        ampl_(amp) {}

  const EncodedEventId &eventId() const { return eventId_; }
  const std::vector<unsigned int> &trackIds() const { return trackIds_; }
  std::vector<unsigned int> &trackIds() { return trackIds_; }  // needed ATM in phase2 digitizer
  const unsigned int trackId() const { return trackIds_[0]; }
  const size_t hitIndex() const { return hitIndex_; }
  const unsigned int tofBin() const { return tofBin_; }
  const size_t hitIndex4ChargeRew() const { return hitInd4CR_; }
  const float getAmpl() const { return ampl_; }

private:
  EncodedEventId eventId_;
  std::vector<unsigned int> trackIds_;
  size_t hitIndex_;
  unsigned int tofBin_;
  size_t hitInd4CR_;
  float ampl_;
};
#endif

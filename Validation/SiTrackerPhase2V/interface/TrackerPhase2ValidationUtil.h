#ifndef _Validation_SiTrackerPhase2V_TrackerPhase2ValidationUtil_h
#define _Validation_SiTrackerPhase2V_TrackerPhase2ValidationUtil_h
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

namespace phase2tkutil {

  bool isPrimary(const SimTrack& simTrk, const PSimHit* simHit);

}  // namespace phase2tkutil
#endif

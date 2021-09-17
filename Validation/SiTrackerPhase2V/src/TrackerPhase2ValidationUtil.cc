#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2ValidationUtil.h"
bool phase2tkutil::isPrimary(const SimTrack& simTrk, const PSimHit* simHit) {
  bool retval = false;
  unsigned int trkId = simTrk.trackId();
  if (trkId != simHit->trackId())
    return retval;
  int vtxIndex = simTrk.vertIndex();
  int ptype = simHit->processType();
  return ((vtxIndex == 0) && (ptype == 0));
}

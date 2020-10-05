#ifndef _Validation_SiTrackerPhase2V_TrackerPhase2ValidationUtil_h
#define _Validation_SiTrackerPhase2V_TrackerPhase2ValidationUtil_h
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include<string>
#include<sstream>

namespace Phase2TkUtil {

bool isPrimary(const SimTrack& simTrk, const PSimHit* simHit){
  bool retval = false;
  unsigned int trkId = simTrk.trackId();
  if (trkId != simHit->trackId())
    return retval;
  int vtxIndex = simTrk.vertIndex();
  int ptype = simHit->processType();
  if ((vtxIndex == 0) && (ptype == 0))
    return true;
  return false;
}

}
#endif

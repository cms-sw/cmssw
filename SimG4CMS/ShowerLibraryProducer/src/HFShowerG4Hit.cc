#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"

#include <iostream>


G4ThreadLocal G4Allocator<HFShowerG4Hit> *fHFShowerG4HitAllocator = 0;

HFShowerG4Hit::HFShowerG4Hit() : theHitId(0), theTrackId(0), theEdep(0),
				 theTime(0) {}

HFShowerG4Hit::HFShowerG4Hit(G4int hitId, G4int tkID, double edep,
			     double time) : theHitId(hitId), theTrackId(tkID),
					    theEdep(edep), theTime(time) {}

HFShowerG4Hit::~HFShowerG4Hit() {}

HFShowerG4Hit::HFShowerG4Hit(const HFShowerG4Hit &right) {
  theHitId   = right.theHitId;
  theTrackId = right.theTrackId;
  theEdep    = right.theEdep;
  theTime    = right.theTime;
  localPos   = right.localPos;
  globalPos  = right.globalPos;
  momDir     = right.momDir;
}

const HFShowerG4Hit& HFShowerG4Hit::operator=(const HFShowerG4Hit &right) {
  theHitId   = right.theHitId;
  theTrackId = right.theTrackId;
  theEdep    = right.theEdep;
  theTime    = right.theTime;
  localPos   = right.localPos;
  globalPos  = right.globalPos;
  momDir     = right.momDir;
  return *this;
}

int HFShowerG4Hit::operator==(const HFShowerG4Hit &right) const {
  return (this==&right) ? 1 : 0;
}

std::ostream& operator<<(std::ostream& os, const HFShowerG4Hit& hit) {
  os << " Data of this HFShowerG4Hit: ID " << hit.hitId() << " Track ID "
     << hit.trackId() << " Edep " << hit.edep() << " Time " << hit.time()
     << " Position (Local) " << hit.localPosition() << ", " << " (Global) "
     << hit.globalPosition() << " Momentum " << hit.primaryMomDir() << "\n";
  return os;
}

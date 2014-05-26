#include "SimG4CMS/ShowerLibraryProducer/interface/FiberG4Hit.h"
#include <iostream>


G4ThreadLocal G4Allocator<FiberG4Hit> *fFiberG4HitAllocator = 0;

FiberG4Hit::FiberG4Hit() : theTowerId(0), theDepth(0), theTrackId(0),
			   theNpe(0), theTime(0), theLogV(0) {
  theHitPos.SetCoordinates(0.,0.,0.);
}

FiberG4Hit::FiberG4Hit(G4LogicalVolume* logVol, G4int tower, G4int depth,
		       G4int tkID) : theTowerId(tower), theDepth(depth), 
				     theTrackId(tkID), theNpe(0), theTime(0), 
				     theLogV(logVol) {
  theHitPos.SetCoordinates(0.,0.,0.);
}

FiberG4Hit::~FiberG4Hit() {}

FiberG4Hit::FiberG4Hit(const FiberG4Hit &right) {
  theTowerId = right.theTowerId;
  theDepth   = right.theDepth;
  theNpe     = right.theNpe;
  theTime    = right.theTime;
  theHitPos  = right.theHitPos;
  theLogV    = right.theLogV;
}

const FiberG4Hit& FiberG4Hit::operator=(const FiberG4Hit &right) {
  theTowerId = right.theTowerId;
  theDepth   = right.theDepth;
  theNpe     = right.theNpe;
  theTime    = right.theTime;
  theHitPos  = right.theHitPos;
  theLogV    = right.theLogV;
  return *this;
}

int FiberG4Hit::operator==(const FiberG4Hit &right) const {
  return (this==&right) ? 1 : 0;
}

std::ostream& operator<<(std::ostream& os, const FiberG4Hit& hit) {
  os << " Data of this FiberG4Hit are:\n"
     << " TowerId ID: " << hit.towerId() << "\n"
     << " Depth     : " << hit.depth() << "\n"
     << " Track ID  : " << hit.trackId() << "\n"
     << " Nb. of Cerenkov Photons : " << hit.npe() << "\n"
     << " Time   :" << hit.time() << " at " << hit.hitPos() << "\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  return os;
}

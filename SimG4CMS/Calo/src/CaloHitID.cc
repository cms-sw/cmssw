//////////////////////////////////////////////////////////////////////////////
// File: CaloHitID.cc
// Description: Identifier for a calorimetric hit
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include <iomanip>

CaloHitID::CaloHitID(uint32_t unitID, double timeSlice, int trackID, uint16_t depth, float tSlice, bool ignoreTkID)
    : timeSliceUnit(tSlice), ignoreTrackID(ignoreTkID), theFineTrackID(-1) {
  setID(unitID, timeSlice, trackID, depth);
}

CaloHitID::CaloHitID(float tSlice, bool ignoreTkID) : timeSliceUnit(tSlice), ignoreTrackID(ignoreTkID) { reset(); }

CaloHitID::CaloHitID(const CaloHitID& id) {
  theUnitID = id.theUnitID;
  theTimeSlice = id.theTimeSlice;
  theTrackID = id.theTrackID;
  theTimeSliceID = id.theTimeSliceID;
  theDepth = id.theDepth;
  timeSliceUnit = id.timeSliceUnit;
  ignoreTrackID = id.ignoreTrackID;
  theFineTrackID = id.theFineTrackID;
}

const CaloHitID& CaloHitID::operator=(const CaloHitID& id) {
  theUnitID = id.theUnitID;
  theTimeSlice = id.theTimeSlice;
  theTrackID = id.theTrackID;
  theTimeSliceID = id.theTimeSliceID;
  theDepth = id.theDepth;
  timeSliceUnit = id.timeSliceUnit;
  ignoreTrackID = id.ignoreTrackID;
  theFineTrackID = id.theFineTrackID;
  return *this;
}

CaloHitID::~CaloHitID() {}

void CaloHitID::setID(uint32_t unitID, double timeSlice, int trackID, uint16_t depth) {
  theUnitID = unitID;
  theTimeSlice = timeSlice;
  theTrackID = trackID;
  theTimeSliceID = (int)(theTimeSlice / timeSliceUnit);
  theDepth = depth;
}

void CaloHitID::reset() {
  theUnitID = 0;
  theTimeSlice = -2 * timeSliceUnit;
  theTrackID = -2;
  theTimeSliceID = (int)(theTimeSlice / timeSliceUnit);
  theDepth = 0;
  theFineTrackID = -1;
}

bool CaloHitID::operator==(const CaloHitID& id) const {
  return ((theUnitID == id.unitID()) && (theTrackID == id.trackID() || ignoreTrackID) &&
          (theTimeSliceID == id.timeSliceID()) && (theDepth == id.depth()) && (fineTrackID() == id.fineTrackID()))
             ? true
             : false;
}

bool CaloHitID::operator<(const CaloHitID& id) const {
  if (theTrackID != id.trackID()) {
    return (theTrackID > id.trackID());
  } else if (fineTrackID() != id.fineTrackID()) {
    return (fineTrackID() > id.fineTrackID());
  } else if (theUnitID != id.unitID()) {
    return (theUnitID > id.unitID());
  } else if (theDepth != id.depth()) {
    return (theDepth > id.depth());
  } else {
    return (theTimeSliceID > id.timeSliceID());
  }
}

bool CaloHitID::operator>(const CaloHitID& id) const {
  if (theTrackID != id.trackID()) {
    return (theTrackID < id.trackID());
  } else if (fineTrackID() != id.fineTrackID()) {
    return (fineTrackID() < id.fineTrackID());
  } else if (theUnitID != id.unitID()) {
    return (theUnitID < id.unitID());
  } else if (theDepth != id.depth()) {
    return (theDepth < id.depth());
  } else {
    return (theTimeSliceID < id.timeSliceID());
  }
}

std::ostream& operator<<(std::ostream& os, const CaloHitID& id) {
  os << "UnitID 0x" << std::hex << id.unitID() << std::dec << " Depth " << std::setw(6) << id.depth() << " Time "
     << std::setw(6) << id.timeSlice() << " TrackID " << std::setw(8) << id.trackID();
  if (id.hasFineTrackID())
    os << " fineTrackID " << id.fineTrackID();
  return os;
}

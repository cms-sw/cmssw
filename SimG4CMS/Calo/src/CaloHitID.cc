///////////////////////////////////////////////////////////////////////////////
// File: CaloHitID.cc
// Description: Identifier for a calorimetric hit
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include <iomanip>

CaloHitID::CaloHitID(uint32_t unitID, double timeSlice, int trackID) {
  setID(unitID, timeSlice, trackID);
}

CaloHitID::CaloHitID() {
  reset();
}

CaloHitID::CaloHitID(const CaloHitID & id) {
  theUnitID      = id.theUnitID;
  theTimeSlice   = id.theTimeSlice;
  theTrackID     = id.theTrackID;
  theTimeSliceID = id.theTimeSliceID;
}

const CaloHitID& CaloHitID::operator=(const CaloHitID & id) {
  theUnitID      = id.theUnitID;
  theTimeSlice   = id.theTimeSlice;
  theTrackID     = id.theTrackID;
  theTimeSliceID = id.theTimeSliceID;

  return *this;
}

CaloHitID::~CaloHitID() {}

void CaloHitID::setID(uint32_t unitID, double timeSlice, int trackID) {
  theUnitID    = unitID;
  theTimeSlice = timeSlice;
  theTrackID   = trackID;
  theTimeSliceID = (int)theTimeSlice;
}

void CaloHitID::reset() {
  theUnitID    = 0;
  theTimeSlice =-2;
  theTrackID   =-2;
  theTimeSliceID = (int)theTimeSlice;
}

bool CaloHitID::operator==(const CaloHitID& id) const {
  return (theUnitID == id.unitID() && theTrackID == id.trackID() &&
	  theTimeSliceID == id.timeSliceID()) ? true : false;
}

bool CaloHitID::operator<(const CaloHitID& id) const {
  if (theTrackID != id.trackID()) {
    return (theTrackID > id.trackID());
  } else if  (theUnitID != id.unitID()) {
    return (theUnitID > id.unitID());
  } else {
    return (theTimeSliceID > id.timeSliceID());
  }
}

bool CaloHitID::operator>(const CaloHitID& id) const {
  if (theTrackID != id.trackID()) {
    return (theTrackID < id.trackID());
  } else if  (theUnitID != id.unitID()) {
    return (theUnitID < id.unitID());
  } else {
    return (theTimeSliceID < id.timeSliceID());
  }
}

ostream& operator<<(ostream& os, const CaloHitID& id) {
  os << "UnitID 0x" << std::hex << id.unitID() << std::dec << " Time " 
     << std::setw(6) << id.timeSlice() << " TrackID " << std::setw(8)
     << id.trackID();
  return os;
}

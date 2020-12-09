#ifndef SimG4CMS_CaloHitID_H
#define SimG4CMS_CaloHitID_H
///////////////////////////////////////////////////////////////////////////////
// File: CaloHitID.h
// HitID class for storing unique identifier of a Calorimetric Hit
///////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <iostream>

class CaloHitID {
public:
  CaloHitID(
      uint32_t unitID, double timeSlice, int trackID, uint16_t depth = 0, float tSlice = 1, bool ignoreTkID = false);
  CaloHitID(float tSlice = 1, bool ignoreTkID = false);
  CaloHitID(const CaloHitID&);
  const CaloHitID& operator=(const CaloHitID&);
  virtual ~CaloHitID();

  uint32_t unitID() const { return theUnitID; }
  int timeSliceID() const { return theTimeSliceID; }
  double timeSlice() const { return theTimeSlice; }
  int trackID() const { return theTrackID; }
  uint16_t depth() const { return theDepth; }
  void setID(uint32_t unitID, double timeSlice, int trackID, uint16_t depth = 0);
  void reset();

  void setTrackID(int trackID) { theTrackID = trackID; }
  bool hasFineTrackID() const { return theFineTrackID != -1; }
  void setFineTrackID(int fineTrackID) { theFineTrackID = fineTrackID; }
  int fineTrackID() const { return hasFineTrackID() ? theFineTrackID : theTrackID; }

  bool operator==(const CaloHitID&) const;
  bool operator<(const CaloHitID&) const;
  bool operator>(const CaloHitID&) const;

private:
  uint32_t theUnitID;
  double theTimeSlice;
  int theTrackID;
  int theTimeSliceID;
  uint16_t theDepth;
  float timeSliceUnit;
  bool ignoreTrackID;
  int theFineTrackID;
};

std::ostream& operator<<(std::ostream&, const CaloHitID&);
#endif

#ifndef SimG4CMS_CaloHitID_H
#define SimG4CMS_CaloHitID_H
///////////////////////////////////////////////////////////////////////////////
// File: CaloHitID.h
// HitID class for storing unique identifier of a Calorimetric Hit
///////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>
#include <iostream>

using namespace std;

class CaloHitID {

public:

  CaloHitID(uint32_t unitID, double timeSlice, int trackID);
  CaloHitID();
  CaloHitID(const CaloHitID&);
  const CaloHitID& operator=(const CaloHitID&);
  virtual ~CaloHitID();

  uint32_t     unitID()      const {return theUnitID;}
  int          timeSliceID() const {return theTimeSliceID;}
  double       timeSlice()   const {return theTimeSlice;}
  int          trackID()     const {return theTrackID;}
  void         setID(unsigned int unitID, double timeSlice, int trackID);
  void         reset();

  bool operator==(const CaloHitID& ) const;
  bool operator<(const CaloHitID& )  const;
  bool operator>(const CaloHitID& )  const;
 
private:

  uint32_t     theUnitID;
  double       theTimeSlice;
  int          theTrackID;
  int          theTimeSliceID;

};

std::ostream& operator<<(std::ostream&, const CaloHitID&);
#endif

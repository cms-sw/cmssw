///////////////////////////////////////////////////////////////////////////////
// File: CaloHitID.h
// HitID class for storing unique identifier of a Calorimetric Hit
///////////////////////////////////////////////////////////////////////////////
#ifndef CaloHitID_H
#define CaloHitID_H

//#include "Utilities/UI/interface/Verbosity.h"
#include <iostream>

using namespace std;

class CaloHitID {

public:

  CaloHitID(unsigned int unitID, double timeSlice, int trackID);
  CaloHitID();
  CaloHitID(const CaloHitID&);
  const CaloHitID& operator=(const CaloHitID&);
  virtual ~CaloHitID();

  unsigned int unitID()      const {return theUnitID;}
  int          timeSliceID() const {return theTimeSliceID;}
  double       timeSlice()   const {return theTimeSlice;}
  int          trackID()     const {return theTrackID;}
  void         setID(unsigned int unitID, double timeSlice, int trackID);
  void         reset();

  bool operator==(const CaloHitID& ) const;
  bool operator<(const CaloHitID& )  const;
  bool operator>(const CaloHitID& )  const;
 
  void         print();
 
private:

  unsigned int theUnitID;
  double       theTimeSlice;
  int          theTrackID;
  int          theTimeSliceID;

  //  static UserVerbosity cout;

};

std::ostream& operator<<(std::ostream&, const CaloHitID&);
#endif

///////////////////////////////////////////////////////////////////////////////
// File: FP420NumberingScheme.h
// Date: 02.2006
// Description: Numbering scheme for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef FP420NumberingScheme_h
#define FP420NumberingScheme_h

#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include <map>



class FP420NumberingScheme {
  
 public:
  FP420NumberingScheme();
  virtual ~FP420NumberingScheme();
  
  virtual unsigned int getUnitID(const G4Step* aStep) const;
  
  // Utilities to get detector levels during a step
  virtual int  detectorLevel(const G4Step*) const;
  virtual void detectorLevel(const G4Step*, int&, int*, G4String*) const;
  
  
  //protected:
  
  static unsigned int packFP420Index(int det, int zside, int station,int plane);
  
  static void unpackFP420Index(const unsigned int& idx, int& det, int& zside, int& station,int& plane);
  
  
  //private:
  //
  //  static UserVerbosity cout;
  
};

#endif

///////////////////////////////////////////////////////////////////////////////
// File: BscNumberingScheme.h
// Date: 02.2006
// Description: Numbering scheme for Bsc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef BscNumberingScheme_h
#define BscNumberingScheme_h

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include "G4ThreeVector.hh"
#include <map>




class BscNumberingScheme {
  
 public:
  BscNumberingScheme();
  virtual ~BscNumberingScheme();
  
  virtual unsigned int getUnitID(const G4Step* aStep) const;
  
  // Utilities to get detector levels during a step
  virtual int  detectorLevel(const G4Step*) const;
  virtual void detectorLevel(const G4Step*, int&, int*, G4String*) const;
  
  
  //protected:
  
  static unsigned int packBscIndex(int det, int zside, int station);
  
  static void unpackBscIndex(const unsigned int& idx);
  
  
  
};

#endif

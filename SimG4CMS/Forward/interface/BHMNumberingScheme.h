#ifndef SimG4CMSForwardBHMNumberingScheme_h
#define SimG4CMSForwardBHMNumberingScheme_h

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include "G4ThreeVector.hh"
#include <map>




class BHMNumberingScheme {
  
public:
  BHMNumberingScheme();
  virtual ~BHMNumberingScheme();
  
  virtual unsigned int getUnitID(const G4Step* aStep) const;
  
  // Utilities to get detector levels during a step
  virtual int  detectorLevel(const G4Step*) const;
  virtual void detectorLevel(const G4Step*, int&, int*, G4String*) const;
  
  
  //protected:
  
  static unsigned int packIndex(int subdet, int zside, int station);
  static void unpackIndex(const unsigned int& idx, int& subdet, int& zside,
			  int& station);
  
};

#endif

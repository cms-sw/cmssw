///////////////////////////////////////////////////////////////////////////////
// File: CaloNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for Calorimeters
///////////////////////////////////////////////////////////////////////////////
#ifndef CaloNumberingScheme_h
#define CaloNumberingScheme_h

#include "G4Step.hh"

class CaloNumberingScheme {

public:
  CaloNumberingScheme(int iv=0);
  virtual ~CaloNumberingScheme(){};
  void    setVerbosity(const int);
	 
  // Utilities to get detector levels during a step
  virtual int  detectorLevel(const G4Step*) const;
  virtual void detectorLevel(const G4Step*, int&, int*, G4String*) const;
    
protected:
  int verbosity;

};

#endif

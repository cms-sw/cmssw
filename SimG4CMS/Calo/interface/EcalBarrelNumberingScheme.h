///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.h
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalBarrelNumberingScheme_h
#define EcalBarrelNumberingScheme_h

#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"

class EcalBarrelNumberingScheme : public CaloNumberingScheme {

public:
  EcalBarrelNumberingScheme();
  ~EcalBarrelNumberingScheme();
	 
  virtual unsigned int getUnitID(const G4Step* aStep) const ;

private:
  //  static UserVerbosity cout;

};

#endif

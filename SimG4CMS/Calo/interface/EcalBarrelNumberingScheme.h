///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.h
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalBarrelNumberingScheme_h
#define EcalBarrelNumberingScheme_h

#include "SimG4CMS/Calo/interface/EcalNumberingScheme.h"

class EcalBarrelNumberingScheme : public EcalNumberingScheme {

public:
  EcalBarrelNumberingScheme(int);
  ~EcalBarrelNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const ;
  virtual float energyInMatrix(int nCellInEta, int nCellInPhi, 
			       int centralEta, int centralPhi, int centralZ,
			       MapType& themap); 

};

#endif

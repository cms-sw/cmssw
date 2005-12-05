///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.h
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalEndcapNumberingScheme_h
#define EcalEndcapNumberingScheme_h

#include "SimG4CMS/Calo/interface/EcalNumberingScheme.h"

class EcalEndcapNumberingScheme : public EcalNumberingScheme {

public:
  EcalEndcapNumberingScheme(int);
  ~EcalEndcapNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const ;
  virtual float energyInMatrix(int nCellInEta, int nCellInPhi, 
			       int centralEta, int centralPhi, int centralZ,
			       MapType& themap); 
};

#endif

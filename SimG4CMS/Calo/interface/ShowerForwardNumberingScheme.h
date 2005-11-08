///////////////////////////////////////////////////////////////////////////////
// File: ShowerForwardNumberingScheme.h
// Description: Numbering scheme for preshower detector
///////////////////////////////////////////////////////////////////////////////
#ifndef ShowerForwardNumberingScheme_h
#define ShowerForwardNumberingScheme_h

#include "SimG4CMS/Calo/interface/EcalNumberingScheme.h"

class ShowerForwardNumberingScheme : public EcalNumberingScheme {

public:
  ShowerForwardNumberingScheme(int);
  ~ShowerForwardNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const;

};

#endif

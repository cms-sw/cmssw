///////////////////////////////////////////////////////////////////////////////
// File: ShowerForwardNumberingScheme.h
// Description: Numbering scheme for preshower detector
///////////////////////////////////////////////////////////////////////////////
#ifndef ShowerForwardNumberingScheme_h
#define ShowerForwardNumberingScheme_h

#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"

class ShowerForwardNumberingScheme : public CaloNumberingScheme {

public:

  ShowerForwardNumberingScheme();
  ~ShowerForwardNumberingScheme();
	 
  virtual unsigned int getUnitID(const G4Step* aStep) const ;

private:


};

#endif

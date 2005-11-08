///////////////////////////////////////////////////////////////////////////////
// File: EcalNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for ECal
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalNumberingScheme_h
#define EcalNumberingScheme_h

#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"
#include <boost/cstdint.hpp>

class EcalNumberingScheme : public CaloNumberingScheme {

public:
  EcalNumberingScheme(int);
  virtual ~EcalNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const = 0;

};

#endif

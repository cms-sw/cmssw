///////////////////////////////////////////////////////////////////////////////
// File: TBHcal04XtalNumberingScheme.h
// Description: Numbering scheme for crystal calorimeter in 2004 test beam
///////////////////////////////////////////////////////////////////////////////
#ifndef TBHcal04XtalNumberingScheme_h
#define TBHcal04XtalNumberingScheme_h

#include "SimG4CMS/Calo/interface/EcalNumberingScheme.h"

class TBHcal04XtalNumberingScheme : public EcalNumberingScheme {

public:
  TBHcal04XtalNumberingScheme(int);
  ~TBHcal04XtalNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const;

};

#endif

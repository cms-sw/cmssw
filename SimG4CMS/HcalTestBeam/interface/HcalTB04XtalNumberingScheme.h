///////////////////////////////////////////////////////////////////////////////
// File: HcalTB04XtalNumberingScheme.h
// Description: Numbering scheme for crystal calorimeter in 2004 test beam
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalTB04XtalNumberingScheme_h
#define HcalTB04XtalNumberingScheme_h

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class HcalTB04XtalNumberingScheme : public EcalNumberingScheme {

public:
  HcalTB04XtalNumberingScheme();
  ~HcalTB04XtalNumberingScheme();
  virtual uint32_t getUnitID(const EcalBaseNumber& baseNumber) const ;

};

#endif

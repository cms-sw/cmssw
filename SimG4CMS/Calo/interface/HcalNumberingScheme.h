///////////////////////////////////////////////////////////////////////////////
// File: HcalNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HCal
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalNumberingScheme_h
#define HcalNumberingScheme_h

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include <boost/cstdint.hpp>

class HcalNumberingScheme : public CaloNumberingScheme {

public:
  HcalNumberingScheme(int);
  virtual ~HcalNumberingScheme();
  virtual uint32_t getUnitID(const HcalNumberingFromDDD::HcalID id);

};

#endif

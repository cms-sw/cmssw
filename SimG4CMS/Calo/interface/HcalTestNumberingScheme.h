///////////////////////////////////////////////////////////////////////////////
// File: HcalTestNumberingScheme.h
// Description: Numbering scheme for hadron calorimeter (detailed for TB)
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalTestNumberingScheme_h
#define HcalTestNumberingScheme_h

#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"

class HcalTestNumberingScheme : public HcalNumberingScheme {

public:
  HcalTestNumberingScheme(int);
  virtual ~HcalTestNumberingScheme();
  virtual uint32_t getUnitID(const HcalNumberingFromDDD::HcalID id);
  static uint32_t  packHcalIndex(int det, int z, int depth, int eta, 
				 int phi, int lay);
  static void      unpackHcalIndex(const uint32_t & idx, int& det, int& z, 
				   int& depth, int& eta, int& phi, int& lay);

};

#endif

#ifndef HcalTestBeam_HcalTB04XtalNumberingScheme_H
#define HcalTestBeam_HcalTB04XtalNumberingScheme_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB04XtalNumberingScheme
//
/**\class HcalTB04XtalNumberingScheme HcalTB04XtalNumberingScheme.h SimG4CMS/HcalTestBeam/interface/HcalTB04XtalNumberingScheme.h
  
 Description:  Numbering scheme for crystal calorimeter in 2004 test beam
  
 Usage: Sets up unique identifier for crystals in 2004 test beam
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu May 18 10:14:34 CEST 2006
//

// system include files

// user include files
#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

class HcalTB04XtalNumberingScheme : public EcalNumberingScheme {
public:
  HcalTB04XtalNumberingScheme();
  ~HcalTB04XtalNumberingScheme() override;
  uint32_t getUnitID(const EcalBaseNumber& baseNumber) const override;
};

#endif

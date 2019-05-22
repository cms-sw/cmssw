#ifndef HcalTestBeam_HcalTB02XtalNumberingScheme_H
#define HcalTestBeam_HcalTB02XtalNumberingScheme_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02XtalNumberingScheme
//
/**\class HcalTB02XtalNumberingScheme HcalTB02XtalNumberingScheme.h SimG4CMS/HcalTestBeam/interface/HcalTB02XtalNumberingScheme.h
  
 Description:  Numbering scheme for the crystal calorimeter in 2002 test beam
  
 Usage: Sets up unique identifier for crystals in 2002 test beam
*/
//
// Original Author:
//         Created:  Fri May 20 10:14:34 CEST 2006
//

// system include files

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02NumberingScheme.h"

class HcalTB02XtalNumberingScheme : public HcalTB02NumberingScheme {
public:
  HcalTB02XtalNumberingScheme();
  ~HcalTB02XtalNumberingScheme() override;
  int getUnitID(const G4Step* aStep) const override;
};

#endif

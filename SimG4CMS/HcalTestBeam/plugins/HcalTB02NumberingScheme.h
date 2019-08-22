#ifndef HcalTestBeam_HcalTB02NumberingScheme_H
#define HcalTestBeam_HcalTB02NumberingScheme_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02NumberingScheme
//
/**\class HcalTB02NumberingScheme HcalTB02NumberingScheme.h SimG4CMS/HcalTestBeam/interface/HcalTB02NumberingScheme.h
  
 Description:  Numbering scheme for hadron calorimeter in 2002 test beam
  
 Usage: Sets up unique identifier for HB towers in 2002 test beam
*/
//
// Original Author:
//         Created:  Fri May 20 10:14:34 CEST 2006
//

// system include files

// user include files
#include "G4Step.hh"

class HcalTB02NumberingScheme {
public:
  HcalTB02NumberingScheme(){};
  virtual ~HcalTB02NumberingScheme(){};
  virtual int getUnitID(const G4Step* aStep) const = 0;
};

#endif

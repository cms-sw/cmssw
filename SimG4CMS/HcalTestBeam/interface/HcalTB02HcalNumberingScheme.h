#ifndef HcalTestBeam_HcalTB02HcalNumberingScheme_H
#define HcalTestBeam_HcalTB02HcalNumberingScheme_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02HcalNumberingScheme
//
/**\class HcalTB02HcalNumberingScheme HcalTB02HcalNumberingScheme.h SimG4CMS/HcalTestBeam/interface/HcalTB02HcalNumberingScheme.h
  
 Description:  Numbering scheme for hadron calorimeter in 2002 test beam
  
 Usage: Sets up unique identifier for HB towers in 2002 test beam
*/
//
// Original Author:  
//         Created:  Fri May 20 10:14:34 CEST 2006
// $Id: HcalTB02HcalNumberingScheme.h,v 1.1 2006/05/23 10:53:29 sunanda Exp $
//
  
// system include files
 
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02NumberingScheme.h"

class HcalTB02HcalNumberingScheme : public HcalTB02NumberingScheme {

public:
  HcalTB02HcalNumberingScheme();
  virtual ~HcalTB02HcalNumberingScheme();
  virtual int getUnitID(const G4Step* aStep) const;

  int getphiScaleF() const { return phiScale;}
  int getetaScaleF() const { return etaScale;}

  int getlayerID(int sID) const;
  int getphiID  (int sID) const;
  int getetaID  (int sID) const;

private:

  int phiScale;
  int etaScale;

};

#endif

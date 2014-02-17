#ifndef HcalTestBeam_HcalTB02SD_H
#define HcalTestBeam_HcalTB02SD_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02SD
//
/**\class HcalTB02SD HcalTB02SD.h SimG4CMS/HcalTestBeam/interface/HcalTB02SD.h
  
 Description:  Stores hits of Test Beam 2002 calorimeters
  
 Usage: Activation is done using the XML file by choosing HcalTB02SensitiveDetector
*/
//
// Original Author:  
//         Created:  Fri May 20 10:14:34 CEST 2006
// $Id: HcalTB02SD.h,v 1.4 2008/05/13 07:18:37 sunanda Exp $
//
  
// system include files
#include <map>
 
// user include files
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02NumberingScheme.h"

#include "G4String.hh"
#include <boost/cstdint.hpp>

class HcalTB02SD : public CaloSD {

public:
  HcalTB02SD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	     edm::ParameterSet const &, const SimTrackManager*);
  virtual ~HcalTB02SD();
  virtual double getEnergyDeposit(G4Step*);
  virtual uint32_t setDetUnitId(G4Step* step);
  void setNumberingScheme(HcalTB02NumberingScheme* scheme);

private:    

  void   initMap(G4String, const DDCompactView &);
  double curve_LY(G4String& , G4StepPoint* ); 
  double crystalLength(G4String);

private:    

  HcalTB02NumberingScheme * numberingScheme;
  bool                      useWeight;
  bool                      useBirk;
  double                    birk1, birk2, birk3;
  std::map<G4String,double> lengthMap;
};

#endif

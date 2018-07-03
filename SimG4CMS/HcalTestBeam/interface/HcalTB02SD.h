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
  HcalTB02SD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &,
	     edm::ParameterSet const &, const SimTrackManager*);
  ~HcalTB02SD() override;
  uint32_t setDetUnitId(const G4Step* step) override;
  void setNumberingScheme(HcalTB02NumberingScheme* scheme);

protected:

  double getEnergyDeposit(const G4Step*) override;

private:    

  void   initMap(const std::string&, const DDCompactView &);
  double curve_LY(const G4String& , const G4StepPoint* ); 
  double crystalLength(const G4String&);

private:    

  HcalTB02NumberingScheme * numberingScheme;
  bool                      useWeight;
  bool                      useBirk;
  double                    birk1, birk2, birk3;
  std::map<G4String,double> lengthMap;
};

#endif

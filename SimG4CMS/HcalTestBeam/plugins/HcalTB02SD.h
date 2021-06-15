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
#include <cstdint>
#include <map>
#include <string>

// user include files
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB02Parameters.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "HcalTB02NumberingScheme.h"

class HcalTB02SD : public CaloSD {
public:
  HcalTB02SD(const std::string&,
             const edm::EventSetup&,
             const SensitiveDetectorCatalog&,
             edm::ParameterSet const&,
             const SimTrackManager*);
  ~HcalTB02SD() override;
  uint32_t setDetUnitId(const G4Step* step) override;
  void setNumberingScheme(HcalTB02NumberingScheme* scheme);

protected:
  double getEnergyDeposit(const G4Step*) override;

private:
  double curve_LY(const std::string&, const G4StepPoint*);
  double crystalLength(const std::string&);

  std::unique_ptr<HcalTB02NumberingScheme> numberingScheme_;
  bool useWeight_, useBirk_;
  double birk1_, birk2_, birk3_;
  const HcalTB02Parameters* hcalTB02Parameters_;
};

#endif

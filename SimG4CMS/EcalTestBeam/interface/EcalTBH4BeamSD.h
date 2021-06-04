#ifndef SimG4CMS_EcalTBH4BeamSD_h
#define SimG4CMS_EcalTBH4BeamSD_h
///////////////////////////////////////////////////////////////////////////////
// File: EcalTBH4BeamSD.h
// Description: Stores hits of TBH4 hodoscope fibers in appropriate
//              container
// Use in your sensitive detector builder:
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "G4String.hh"
#include <map>

class EcalBaseNumber;

class EcalTBH4BeamSD : public CaloSD {
public:
  EcalTBH4BeamSD(const std::string &,
                 const edm::EventSetup &,
                 const SensitiveDetectorCatalog &,
                 edm::ParameterSet const &,
                 const SimTrackManager *);
  ~EcalTBH4BeamSD() override;
  uint32_t setDetUnitId(const G4Step *step) override;
  void setNumberingScheme(EcalNumberingScheme *scheme);

protected:
  double getEnergyDeposit(const G4Step *) override;

private:
  void getBaseNumber(const G4Step *aStep);
  EcalNumberingScheme *numberingScheme;
  bool useWeight;
  bool useBirk;
  double birk1, birk2, birk3;
  EcalBaseNumber theBaseNumber;
};

#endif  // EcalTBH4BeamSD_h

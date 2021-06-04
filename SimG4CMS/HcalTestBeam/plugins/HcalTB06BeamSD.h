#ifndef SimG4CMS_HcalTB06BeamSD_h
#define SimG4CMS_HcalTB06BeamSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HcalTB06BeamSD.h
// Description: Stores hits of Beam counters for H2 TB06 in appropriate
//              containers
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParameters.h"
#include "G4String.hh"

#include <string>

class G4Step;
class G4Material;

class HcalTB06BeamSD : public CaloSD {
public:
  HcalTB06BeamSD(const std::string &,
                 const edm::EventSetup &,
                 const SensitiveDetectorCatalog &,
                 edm::ParameterSet const &,
                 const SimTrackManager *);
  ~HcalTB06BeamSD() override;
  uint32_t setDetUnitId(const G4Step *step) override;

protected:
  double getEnergyDeposit(const G4Step *) override;

private:
  bool isItWireChamber(const std::string &);

  bool useBirk_;
  double birk1_, birk2_, birk3_;
  const HcalTB06BeamParameters *hcalBeamPar_;
};

#endif  // HcalTB06BeamSD_h

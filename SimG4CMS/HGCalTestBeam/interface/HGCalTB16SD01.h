#ifndef SimG4CMS_HGCalTestBeam_HGCalTB16SD01_H
#define SimG4CMS_HGCalTestBeam_HGCalTB16SD01_H 1

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4Material.hh"
#include "G4Step.hh"

#include <string>

class HGCalTB16SD01 : public CaloSD {
public:
  HGCalTB16SD01(const std::string&, const SensitiveDetectorCatalog&, edm::ParameterSet const&, const SimTrackManager*);
  ~HGCalTB16SD01() override = default;
  uint32_t setDetUnitId(const G4Step* step) override;
  static uint32_t packIndex(int det, int lay, int x, int y);
  static void unpackIndex(const uint32_t& idx, int& det, int& lay, int& x, int& y);

protected:
  double getEnergyDeposit(const G4Step*) override;

private:
  void initialize(const G4StepPoint* point);

  std::string matName_;
  bool useBirk_;
  double birk1_, birk2_, birk3_;
  bool initialize_;
  G4Material* matScin_;
};
#endif

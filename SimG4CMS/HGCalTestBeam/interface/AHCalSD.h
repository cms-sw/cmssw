#ifndef SimG4CMS_HGCalTestBeam_AHCalSD_H
#define SimG4CMS_HGCalTestBeam_AHCalSD_H 1

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"

#include <string>

class AHCalSD : public CaloSD {
public:
  AHCalSD(const std::string&, const SensitiveDetectorCatalog&, edm::ParameterSet const&, const SimTrackManager*);
  ~AHCalSD() override = default;
  uint32_t setDetUnitId(const G4Step* step) override;
  bool unpackIndex(const uint32_t& idx, int& row, int& col, int& depth);

protected:
  double getEnergyDeposit(const G4Step*) override;
  bool filterHit(CaloG4Hit*, double) override;

private:
  bool useBirk;
  double birk1, birk2, birk3, betaThr;
  double eminHit;
};
#endif

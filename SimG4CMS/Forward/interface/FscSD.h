#ifndef SimG4CMSForward_FscSD_h
#define SimG4CMSForward_FscSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

class SimTrackManager;

//-------------------------------------------------------------------

class FscSD : public CaloSD {
public:
  FscSD(const std::string &, const SensitiveDetectorCatalog &, edm::ParameterSet const &, const SimTrackManager *);
  ~FscSD() override = default;

  uint32_t setDetUnitId(const G4Step *) override;

protected:
  double getEnergyDeposit(const G4Step *) override;

private:
  int verbn_;
  bool useBirk_;
  double birk1_, birk2_, birk3_;
};

#endif  // FscSD_h

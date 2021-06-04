#ifndef Forward_PltSD_h
#define Forward_PltSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

class G4Step;
class SimTrackManager;

class PltSD : public TimingSD {
public:
  PltSD(const std::string &,
        const edm::EventSetup &,
        const SensitiveDetectorCatalog &,
        edm::ParameterSet const &,
        const SimTrackManager *);
  ~PltSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

protected:
  bool checkHit(const G4Step *, BscG4Hit *) override;

private:
  double energyCut;
  double energyHistoryCut;
};

#endif

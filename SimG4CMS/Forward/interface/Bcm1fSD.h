#ifndef Forward_Bcm1fSD_h
#define Forward_Bcm1fSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

class SimTrackManager;
class G4Step;

class Bcm1fSD : public TimingSD {
public:
  Bcm1fSD(const std::string &,
          const edm::EventSetup &,
          const SensitiveDetectorCatalog &,
          edm::ParameterSet const &,
          const SimTrackManager *);
  ~Bcm1fSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

protected:
  bool checkHit(const G4Step *, BscG4Hit *) override;

private:
  float energyCut;
  float energyHistoryCut;
};

#endif

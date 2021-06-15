#ifndef SimG4CMSForward_BscSD_h
#define SimG4CMSForward_BscSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

class SimTrackManager;
class BscNumberingScheme;

//-------------------------------------------------------------------

class BscSD : public TimingSD {
public:
  BscSD(const std::string &,
        const edm::EventSetup &,
        const SensitiveDetectorCatalog &,
        edm::ParameterSet const &,
        const SimTrackManager *);

  ~BscSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

private:
  BscNumberingScheme *numberingScheme;
};

#endif  // BscSD_h

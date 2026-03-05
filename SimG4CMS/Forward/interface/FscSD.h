#ifndef SimG4CMSForward_FscSD_h
#define SimG4CMSForward_FscSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

class SimTrackManager;

//-------------------------------------------------------------------

class FscSD : public TimingSD {
public:
  FscSD(const std::string &, const SensitiveDetectorCatalog &, edm::ParameterSet const &, const SimTrackManager *);

  ~FscSD() override;

  uint32_t setDetUnitId(const G4Step *) override;
};

#endif  // FscSD_h

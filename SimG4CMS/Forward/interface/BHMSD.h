#ifndef SimG4CMSForward_BHMSD_h
#define SimG4CMSForward_BHMSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include <string>

class SimTrackManager;
class G4Step;

//-------------------------------------------------------------------

class BHMSD : public TimingSD {
public:
  BHMSD(const std::string &, const SensitiveDetectorCatalog &, edm::ParameterSet const &, const SimTrackManager *);

  ~BHMSD() override;

  uint32_t setDetUnitId(const G4Step *) override;
};

#endif

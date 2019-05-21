#ifndef SimG4CMSForward_BHMSD_h
#define SimG4CMSForward_BHMSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"

#include <string>

class SimTrackManager;
class G4Step;
class BHMNumberingScheme;

//-------------------------------------------------------------------

class BHMSD : public TimingSD {
public:
  BHMSD(const std::string &,
        const DDCompactView &,
        const SensitiveDetectorCatalog &,
        edm::ParameterSet const &,
        const SimTrackManager *);

  ~BHMSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

private:
  BHMNumberingScheme *numberingScheme;
};

#endif

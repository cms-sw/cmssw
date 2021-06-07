#ifndef SimG4Core_SensitiveDetector_AttachSD_h
#define SimG4Core_SensitiveDetector_AttachSD_h

#include <vector>

namespace edm {
  class EventSetup;
  class ParameterSet;
}  // namespace edm

class SensitiveDetectorCatalog;
class SensitiveTkDetector;
class SensitiveCaloDetector;
class SimActivityRegistry;
class SimTrackManager;

namespace sim {
  std::pair<std::vector<SensitiveTkDetector *>, std::vector<SensitiveCaloDetector *> > attachSD(
      const edm::EventSetup &,
      const SensitiveDetectorCatalog &,
      edm::ParameterSet const &,
      const SimTrackManager *,
      SimActivityRegistry &reg);
};

#endif

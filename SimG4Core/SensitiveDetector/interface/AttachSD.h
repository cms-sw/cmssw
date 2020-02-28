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

class AttachSD {
public:
  AttachSD();
  ~AttachSD();

  std::pair<std::vector<SensitiveTkDetector *>, std::vector<SensitiveCaloDetector *> > create(
      const edm::EventSetup &,
      const SensitiveDetectorCatalog &,
      edm::ParameterSet const &,
      const SimTrackManager *,
      SimActivityRegistry &reg) const;
};

#endif

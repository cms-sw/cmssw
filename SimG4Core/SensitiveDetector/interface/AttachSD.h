#ifndef SimG4Core_SensitiveDetector_AttachSD_h
#define SimG4Core_SensitiveDetector_AttachSD_h

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace edm {
  class EventSetup;
  class ParameterSet;
}  // namespace edm

class SensitiveDetectorCatalog;
class SensitiveTkDetector;
class SensitiveCaloDetector;
class SensitiveDetectorMakerBase;
class SimActivityRegistry;
class SimTrackManager;

namespace sim {
  std::pair<std::vector<SensitiveTkDetector *>, std::vector<SensitiveCaloDetector *>> attachSD(
      const std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>> &,
      const edm::EventSetup &,
      const SensitiveDetectorCatalog &,
      edm::ParameterSet const &,
      const SimTrackManager *,
      SimActivityRegistry &reg);
};

#endif

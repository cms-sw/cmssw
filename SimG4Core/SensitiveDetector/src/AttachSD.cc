#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/sensitiveDetectorMakers.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <sstream>
#include <algorithm>

std::pair<std::vector<SensitiveTkDetector*>, std::vector<SensitiveCaloDetector*>> sim::attachSD(
    const std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>>& makers,
    const edm::EventSetup& es,
    const SensitiveDetectorCatalog& clg,
    edm::ParameterSet const& p,
    const SimTrackManager* man,
    SimActivityRegistry& reg) {
  std::pair<std::vector<SensitiveTkDetector*>, std::vector<SensitiveCaloDetector*>> detList;
  const std::vector<std::string_view>& rouNames = clg.readoutNames();

  edm::LogVerbatim("SimG4CoreSensitiveDetector") << " AttachSD: Initialising " << rouNames.size() << " SDs";
  for (auto& rname : rouNames) {
    std::string_view className = clg.className({rname.data(), rname.size()});
    auto makerItr = makers.find(std::string(className.data(), className.size()));
    if (makerItr == makers.end()) {
      throw cms::Exception("MissingSensitiveDetector")
          << "Asked to use " << className << " sensitive detector but it was not in the list of preloaded ones.";
    }

    std::unique_ptr<SensitiveDetector> sd{makerItr->second->make({rname.data(), rname.size()}, es, clg, p, man, reg)};

    std::stringstream ss;
    ss << " AttachSD: created a " << className << " with name " << rname;

    if (sd->isCaloSD()) {
      detList.second.push_back(static_cast<SensitiveCaloDetector*>(sd.release()));
      ss << " + calo SD";
    } else {
      detList.first.push_back(static_cast<SensitiveTkDetector*>(sd.release()));
      ss << " + tracking SD";
    }
    edm::LogVerbatim("SimG4CoreSensitiveDetector") << ss.str();
  }
  return detList;
}

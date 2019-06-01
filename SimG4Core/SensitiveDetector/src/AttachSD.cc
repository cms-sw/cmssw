#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <sstream>

AttachSD::AttachSD() {}

AttachSD::~AttachSD() {}

std::pair<std::vector<SensitiveTkDetector*>, std::vector<SensitiveCaloDetector*> > AttachSD::create(
    const DDCompactView& cpv,
    const SensitiveDetectorCatalog& clg,
    edm::ParameterSet const& p,
    const SimTrackManager* man,
    SimActivityRegistry& reg) const {
  std::pair<std::vector<SensitiveTkDetector*>, std::vector<SensitiveCaloDetector*> > detList;
  const std::vector<std::string>& rouNames = clg.readoutNames();
  edm::LogVerbatim("SimG4CoreSensitiveDetector") << " AttachSD: Initialising " << rouNames.size() << " SDs";
  for (auto& rname : rouNames) {
    std::string className = clg.className(rname);
    std::unique_ptr<SensitiveDetectorMakerBase> temp{SensitiveDetectorPluginFactory::get()->create(className)};

    std::unique_ptr<SensitiveDetector> sd{temp->make(rname, cpv, clg, p, man, reg)};

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

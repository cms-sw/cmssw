// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "SimG4CMS/PPS/interface/TotemRPSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RomanPotSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit RomanPotSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {}
  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<TotemRPSD>(iname, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }
};

typedef TotemRPSD RomanPotSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(RomanPotSensitiveDetectorBuilder, RomanPotSensitiveDetector);

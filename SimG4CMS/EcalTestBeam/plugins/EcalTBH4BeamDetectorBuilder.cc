// system include files
#include <string>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "SimG4CMS/EcalTestBeam/interface/EcalTBH4BeamSD.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalTBH4BeamDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit EcalTBH4BeamDetectorBuilder(const edm::ParameterSet& p, edm::ConsumesCollector cc) {}

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<EcalTBH4BeamSD>(iname, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }
};

typedef EcalTBH4BeamSD EcalTBH4BeamDetector;
DEFINE_SENSITIVEDETECTORBUILDER(EcalTBH4BeamDetectorBuilder, EcalTBH4BeamDetector);

// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "SimG4CMS/HGCalTestBeam/interface/HGCalTB16SD01.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HGCalTB16SensitiveDetector01Builder : public SensitiveDetectorMakerBase {
public:
  explicit HGCalTB16SensitiveDetector01Builder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {}
  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<HGCalTB16SD01>(iname, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }
};

typedef HGCalTB16SD01 HGCalTB1601SensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HGCalTB16SensitiveDetector01Builder, HGCalTB1601SensitiveDetector);

// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "SimG4CMS/Forward/interface/BHMSD.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class BHMSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit BHMSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {}

  void beginRun(const edm::EventSetup& es) final {}

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<BHMSD>(iname, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }
};

typedef BHMSD BHMSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(BHMSensitiveDetectorBuilder, BHMSensitiveDetector);

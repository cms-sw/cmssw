// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/HFChamberSD.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HFChamberSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HFChamberSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {}

  void beginRun(const edm::EventSetup& es) final {}
  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<HFChamberSD>(iname, clg, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }
};

typedef HFChamberSD HFChamberSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HFChamberSensitiveDetectorBuilder, HFChamberSensitiveDetector);

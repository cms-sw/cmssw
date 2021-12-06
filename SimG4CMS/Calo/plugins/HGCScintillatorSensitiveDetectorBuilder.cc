
// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/Calo/interface/HGCScintSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HGCScintillatorSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HGCScintillatorSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : hgcToken_{cc.esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{"", "HGCalHEScintillatorSensitive"})} {}

  void beginRun(const edm::EventSetup& es) final { hgcons_ = es.getHandle(hgcToken_); }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto hgc = ((iname.find("HitsHEback") != std::string::npos) && hgcons_.isValid()) ? hgcons_.product() : nullptr;
    auto sd = std::make_unique<HGCScintSD>(iname, hgc, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgcToken_;
  edm::ESHandle<HGCalDDDConstants> hgcons_;
};

typedef HGCScintSD HGCScintillatorSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HGCScintillatorSensitiveDetectorBuilder, HGCScintillatorSensitiveDetector);

// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DreamSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit DreamSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {
    fromDD4Hep_ = p.getParameter<bool>("g4GeometryDD4hepSource");
    if (fromDD4Hep_)
      cpvTokenDD4Hep_ = cc.esConsumes<edm::Transition::BeginRun>();
    else
      cpvTokenDDD_ = cc.esConsumes<edm::Transition::BeginRun>();
  }

  void beginRun(const edm::EventSetup& es) final {
    if (fromDD4Hep_) {
      cpvDD4Hep_ = &es.getData(cpvTokenDD4Hep_);
    } else {
      cpvDDD_ = &es.getData(cpvTokenDDD_);
    }
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<DreamSD>(iname, cpvDDD_, cpvDD4Hep_, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  bool fromDD4Hep_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  const DDCompactView* cpvDDD_;
  const cms::DDCompactView* cpvDD4Hep_;
};

typedef DreamSD DreamSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(DreamSensitiveDetectorBuilder, DreamSensitiveDetector);

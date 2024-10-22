// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimG4CMS/Tracker/interface/TkAccumulatingSensitiveDetector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TkAccumulatingSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit TkAccumulatingSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : geomdet_{nullptr}, geomdetToken_{cc.esConsumes<edm::Transition::BeginRun>()} {}

  void beginRun(const edm::EventSetup& es) final { geomdet_ = &es.getData(geomdetToken_); }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<TkAccumulatingSensitiveDetector>(iname, geomdet_, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const GeometricDet* geomdet_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomdetToken_;
};

DEFINE_SENSITIVEDETECTORBUILDER(TkAccumulatingSensitiveDetectorBuilder, TkAccumulatingSensitiveDetector);

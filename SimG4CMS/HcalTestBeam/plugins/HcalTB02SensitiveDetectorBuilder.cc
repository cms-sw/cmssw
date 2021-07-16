// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/HcalTestBeamData/interface/HcalTB02Parameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/HcalTestBeam/plugins/HcalTB02SD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalTB02SensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HcalTB02SensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : ebParToken_{cc.esConsumes<HcalTB02Parameters, IdealGeometryRecord, edm::Transition::BeginRun>(
            edm::ESInputTag{"", "EcalHitsEB"})},
        hcParToken_{cc.esConsumes<HcalTB02Parameters, IdealGeometryRecord, edm::Transition::BeginRun>(
            edm::ESInputTag{"", "HcalHits"})},
        ebPar_{nullptr},
        hcPar_{nullptr} {}

  void beginRun(const edm::EventSetup& es) final {
    ebPar_ = &es.getData(ebParToken_);
    hcPar_ = &es.getData(hcParToken_);
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto par = ((iname == "EcalHitsEB") ? ebPar_ : ((iname == "HcalHits") ? hcPar_ : nullptr));
    auto sd = std::make_unique<HcalTB02SD>(iname, par, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<HcalTB02Parameters, IdealGeometryRecord> ebParToken_;
  const edm::ESGetToken<HcalTB02Parameters, IdealGeometryRecord> hcParToken_;
  const HcalTB02Parameters* ebPar_;
  const HcalTB02Parameters* hcPar_;
};

typedef HcalTB02SD HcalTB02SensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HcalTB02SensitiveDetectorBuilder, HcalTB02SensitiveDetector);

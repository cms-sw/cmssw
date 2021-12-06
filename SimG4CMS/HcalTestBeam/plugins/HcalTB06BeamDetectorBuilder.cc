// system include files

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06BeamSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalTB06BeamDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HcalTB06BeamDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : hcParToken_{cc.esConsumes<edm::Transition::BeginRun>()}, hcPar_{nullptr} {}

  void beginRun(const edm::EventSetup& es) final { hcPar_ = &es.getData(hcParToken_); }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<HcalTB06BeamSD>(iname, hcPar_, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<HcalTB06BeamParameters, IdealGeometryRecord> hcParToken_;
  const HcalTB06BeamParameters* hcPar_;
};

typedef HcalTB06BeamSD HcalTB06BeamDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HcalTB06BeamDetectorBuilder, HcalTB06BeamDetector);

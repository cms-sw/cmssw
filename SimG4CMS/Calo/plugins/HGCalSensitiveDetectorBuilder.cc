
// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/Calo/interface/HGCalSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HGCalSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HGCalSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : hgcEEToken_{cc.esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{"", "HGCalEESensitive"})},
        hgcHEToken_{cc.esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{"", "HGCalHESiliconSensitive"})} {
    edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
    int num = m_HGC.getUntrackedParameter<int>("UseDetector");
    doEE_ = ((num % 2) == 1);
    doHE_ = (((num / 2) % 2) == 1);
  }

  void beginRun(const edm::EventSetup& es) final {
    if (doEE_)
      hgcalEE_ = es.getHandle(hgcEEToken_);
    if (doHE_)
      hgcalHE_ = es.getHandle(hgcHEToken_);
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto hgc =
        (((iname.find("HitsEE") != std::string::npos) && doEE_ && hgcalEE_.isValid())
             ? hgcalEE_.product()
             : (((iname.find("HitsHEfront") != std::string::npos) && doHE_ && hgcalHE_.isValid()) ? hgcalHE_.product()
                                                                                                  : nullptr));
    auto sd = std::make_unique<HGCalSD>(iname, hgc, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgcEEToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hgcHEToken_;
  bool doEE_, doHE_;
  edm::ESHandle<HGCalDDDConstants> hgcalEE_, hgcalHE_;
};

typedef HGCalSD HGCalSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HGCalSensitiveDetectorBuilder, HGCalSensitiveDetector);

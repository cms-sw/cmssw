
// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "CondFormats/GeometryObjects/interface/EcalSimulationParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/Calo/interface/ECalSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit EcalSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {
    edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
    num_ = m_EC.getUntrackedParameter<int>("Detectors");
    for (int k = 0; k < num_; ++k)
      ecToken_.emplace_back(cc.esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{"", name_[k]}));
  }

  void beginRun(const edm::EventSetup& es) final {
    for (const auto& token : ecToken_)
      ecpar_.emplace_back(es.getHandle(token));
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    int k = static_cast<int>(std::find(name_, name_ + num_, iname) - name_);
    auto const& par = ((k < num_) && ecpar_[k].isValid()) ? ecpar_[k].product() : nullptr;
    auto sd = std::make_unique<ECalSD>(iname, par, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  static constexpr unsigned int nameSize_ = 3;
  const std::string name_[nameSize_] = {"EcalHitsEB", "EcalHitsEE", "EcalHitsES"};
  int num_;
  std::vector<edm::ESGetToken<EcalSimulationParameters, IdealGeometryRecord>> ecToken_;
  std::vector<edm::ESHandle<EcalSimulationParameters>> ecpar_;
};

typedef ECalSD EcalSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(EcalSensitiveDetectorBuilder, EcalSensitiveDetector);

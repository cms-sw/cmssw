
// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/Calo/interface/HGCSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HGCSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HGCSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc) {
    edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
    num_ = m_HGC.getUntrackedParameter<int>("Detectors");
    for (int k = 0; k < num_; ++k) {
      hgcToken_.emplace_back(cc.esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{"", name0_[k]}));
      edm::LogVerbatim("HGCSim") << "HGCSensitiveDetectorBuilder::Initailize Token[" << k << "] for " << name0_[k];
    }
  }

  void beginRun(const edm::EventSetup& es) final {
    for (const auto& token : hgcToken_)
      hgcons_.emplace_back(es.getHandle(token));
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    const HGCalTBDDDConstants* hgc = nullptr;
    for (int k = 0; k < num_; ++k) {
      if (iname.find(name1_[k]) != std::string::npos) {
        if (hgcons_[k].isValid())
          hgc = hgcons_[k].product();
        break;
      }
    }
    auto sd = std::make_unique<HGCSD>(iname, hgc, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  static constexpr unsigned int nameSize_ = 3;
  const std::string name0_[nameSize_] = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  const std::string name1_[nameSize_] = {"HitsEE", "HitsHEfront", "HitsHEback"};
  int num_;
  std::vector<edm::ESGetToken<HGCalTBDDDConstants, IdealGeometryRecord>> hgcToken_;
  std::vector<edm::ESHandle<HGCalTBDDDConstants>> hgcons_;
};

typedef HGCSD HGCSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HGCSensitiveDetectorBuilder, HGCSensitiveDetector);

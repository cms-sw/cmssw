
// system include files
#include <string>
#include <vector>

// user include files
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorMakerBase.h"
#include "SimG4Core/Notification/interface/SimActivityRegistryEnroller.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

#include "CondFormats/DataRecord/interface/HBHEDarkeningRecord.h"
#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalSensitiveDetectorBuilder : public SensitiveDetectorMakerBase {
public:
  explicit HcalSensitiveDetectorBuilder(edm::ParameterSet const& p, edm::ConsumesCollector cc)
      : hdscToken_{cc.esConsumes<edm::Transition::BeginRun>()},
        hdrcToken_{cc.esConsumes<edm::Transition::BeginRun>()},
        hscsToken_{cc.esConsumes<edm::Transition::BeginRun>()},
        hbdkToken_{
            cc.esConsumes<HBHEDarkening, HBHEDarkeningRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", "HB"})},
        hedkToken_{
            cc.esConsumes<HBHEDarkening, HBHEDarkeningRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", "HE"})},
        hcalDDSim_{nullptr},
        hcalDDRec_{nullptr},
        hcalSimConstants_{nullptr},
        hbDarkening_{nullptr},
        heDarkening_{nullptr} {
    edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HCalSD");
    agingFlagHB_ = m_HC.getParameter<bool>("HBDarkening");
    agingFlagHE_ = m_HC.getParameter<bool>("HEDarkening");
    forTBHC_ = m_HC.getUntrackedParameter<bool>("ForTBHCAL", false);
    forTBH2_ = m_HC.getUntrackedParameter<bool>("ForTBH2", false);
  }

  void beginRun(const edm::EventSetup& es) final {
    hcalDDSim_ = &es.getData(hdscToken_);
    if ((!forTBHC_) && (!forTBH2_))
      hcalDDRec_ = &es.getData(hdrcToken_);
    edm::ESHandle<HcalSimulationConstants> hscs = es.getHandle(hscsToken_);
    if (hscs.isValid())
      hcalSimConstants_ = hscs.product();
    else
      edm::LogWarning("HcalSim") << "HcalSensitiveDetectorBuilder does not find record for HcalSimulationConstants";
    if (agingFlagHB_) {
      edm::ESHandle<HBHEDarkening> hbdark = es.getHandle(hbdkToken_);
      if (hbdark.isValid())
        hbDarkening_ = hbdark.product();
      else
        edm::LogVerbatim("HcalSim") << "HcalSensitiveDetectorBuilder does not find record for HBDarkening";
    }
    if (agingFlagHE_) {
      edm::ESHandle<HBHEDarkening> hedark = es.getHandle(hedkToken_);
      if (hedark.isValid())
        heDarkening_ = hedark.product();
      else
        edm::LogVerbatim("HcalSim") << "HcalSensitiveDetectorBuilder does not find record for HEDarkening";
    }
  }

  std::unique_ptr<SensitiveDetector> make(const std::string& iname,
                                          const SensitiveDetectorCatalog& clg,
                                          const edm::ParameterSet& p,
                                          const SimTrackManager* man,
                                          SimActivityRegistry& reg) const final {
    auto sd = std::make_unique<HCalSD>(
        iname, hcalDDSim_, hcalDDRec_, hcalSimConstants_, hbDarkening_, heDarkening_, clg, p, man);
    SimActivityRegistryEnroller::enroll(reg, sd.get());
    return sd;
  }

private:
  const edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> hdscToken_;
  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> hdrcToken_;
  const edm::ESGetToken<HcalSimulationConstants, HcalSimNumberingRecord> hscsToken_;
  const edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> hbdkToken_;
  const edm::ESGetToken<HBHEDarkening, HBHEDarkeningRecord> hedkToken_;
  const HcalDDDSimConstants* hcalDDSim_;
  const HcalDDDRecConstants* hcalDDRec_;
  const HcalSimulationConstants* hcalSimConstants_;
  const HBHEDarkening* hbDarkening_;
  const HBHEDarkening* heDarkening_;
  bool agingFlagHB_, agingFlagHE_;
  bool forTBHC_, forTBH2_;
};

typedef HCalSD HcalSensitiveDetector;
DEFINE_SENSITIVEDETECTORBUILDER(HcalSensitiveDetectorBuilder, HcalSensitiveDetector);

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <string>
#include <vector>

class HGCalSimHitDump : public edm::one::EDAnalyzer<> {
public:
  HGCalSimHitDump(const edm::ParameterSet& ps);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const std::string g4Label_;
  const std::vector<std::string> hitLab_;
  const std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> toksCalo_;
  const std::vector<int> types_;
  const int maxEvent_;
  int kount_;
};

HGCalSimHitDump::HGCalSimHitDump(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("ModuleLabel")),
      hitLab_(ps.getParameter<std::vector<std::string>>("HitCollections")),
      toksCalo_{edm::vector_transform(
          hitLab_,
          [this](const std::string& name) { return consumes<edm::PCaloHitContainer>(edm::InputTag{g4Label_, name}); })},
      types_(ps.getParameter<std::vector<int>>("CollectionTypes")),
      maxEvent_(ps.getParameter<int>("MaxEvent")),
      kount_(0) {
  edm::LogVerbatim("HitStudy") << "Module Label: " << g4Label_ << "   with " << hitLab_.size()
                               << " collections and maxEvent = " << maxEvent_;
  for (unsigned int k = 0; k < hitLab_.size(); ++k)
    edm::LogVerbatim("HitStudy") << "[" << k << "] Type " << types_[k] << " Label " << hitLab_[k];
}

void HGCalSimHitDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> coll = {"HGCHitsEE",
                                   "CalibrationHGCHitsEE",
                                   "HGCHitsHEfront",
                                   "CalibrationHGCHitsHEfront",
                                   "HGCHitsHEback",
                                   "CalibrationHGCHitsHEback"};
  std::vector<int> type = {0, 0, 0, 0, 1, 1};
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::vector<std::string>>("HitCollections", coll);
  desc.add<std::vector<int>>("CollectionTypes", type);
  desc.add<int>("MaxEvent", 10);
  descriptions.add("hgcalSimHitDump", desc);
}

void HGCalSimHitDump::analyze(const edm::Event& e, const edm::EventSetup&) {
  ++kount_;
  edm::LogVerbatim("HitStudy") << "[" << kount_ << "] Run = " << e.id().run() << " Event = " << e.id().event();

  if ((kount_ <= maxEvent_) || (maxEvent_ <= 0)) {
    for (unsigned int k = 0; k < toksCalo_.size(); ++k) {
      const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(toksCalo_[k]);
      if (hitsCalo.isValid()) {
        edm::LogVerbatim("HitStudy") << "HGCalSimHitDump: Input " << hitsCalo->size() << " hits of type " << hitLab_[k]
                                     << ":" << types_[k];
        unsigned int i(0);
        for (auto const& hit : *hitsCalo) {
          double edep = hit.energy();
          double time = hit.time();
          unsigned int id = hit.id();
          if (types_[k] == 0)
            edm::LogVerbatim("HitStudy") << "[" << i << "] " << HGCSiliconDetId(id) << " E" << edep << " T " << time;
          else
            edm::LogVerbatim("HitStudy") << "[" << i << "] " << HGCScintillatorDetId(id) << " E" << edep << " T "
                                         << time;
          ++i;
        }
      } else {
        edm::LogVerbatim("HitStudy") << "HGCalSimHitDump: No Hits found for type " << hitLab_[k] << ":" << types_[k];
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalSimHitDump);

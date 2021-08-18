#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <map>
#include <string>
#include <vector>

class HGcalHitIdCheck : public edm::one::EDAnalyzer<> {
public:
  HGcalHitIdCheck(const edm::ParameterSet& ps);
  ~HGcalHitIdCheck() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  const std::string g4Label_, caloHitSource_, nameSense_, nameDetector_;
  const int verbosity_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGcalHitIdCheck::HGcalHitIdCheck(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("moduleLabel")),
      caloHitSource_(ps.getParameter<std::string>("caloHitSource")),
      nameSense_(ps.getParameter<std::string>("nameSense")),
      nameDetector_(ps.getParameter<std::string>("nameDevice")),
      verbosity_(ps.getParameter<int>("Verbosity")),
      tok_calo_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, caloHitSource_))),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HitStudy") << "Test Hit ID for " << nameDetector_ << " using SimHits for " << nameSense_
                               << " with module Label: " << g4Label_ << "   Hits: " << caloHitSource_;
}

void HGcalHitIdCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("moduleLabel", "g4SimHits");
  desc.add<std::string>("caloHitSource", "HGCHitsEE");
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<std::string>("nameDevice", "HGCal EE");
  desc.add<int>("Verbosity", 0);
  descriptions.add("hgcalHitIdCheckEE", desc);
}

void HGcalHitIdCheck::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  if (verbosity_ > 0)
    edm::LogVerbatim("HitStudy") << "Run = " << e.id().run() << " Event = " << e.id().event();

  // get hcalGeometry
  const HGCalGeometry* geom = &iS.getData(geomToken_);
  const std::vector<DetId>& validIds = geom->getValidDetIds();

  edm::Handle<edm::PCaloHitContainer> hitsCalo;
  e.getByToken(tok_calo_, hitsCalo);
  bool getHits = (hitsCalo.isValid());
  uint32_t nhits = (getHits) ? hitsCalo->size() : 0;
  uint32_t good(0), all(0);
  if (verbosity_ > 1)
    edm::LogVerbatim("HitStudy") << "HGcalHitIdCheck: Input flags Hits " << getHits << " with " << nhits << " hits";

  if (getHits) {
    std::vector<PCaloHit> hits;
    hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
    if (!hits.empty()) {
      for (auto hit : hits) {
        ++all;
        DetId id(hit.id());
        if (std::find(validIds.begin(), validIds.end(), id) != validIds.end()) {
          ++good;
          if (verbosity_ > 2) {
            if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
              edm::LogVerbatim("HitStudy") << "Hit[" << all << ":" << good << "]" << HGCSiliconDetId(id);
            } else if (id.det() == DetId::HGCalHSc) {
              edm::LogVerbatim("HitStudy") << "Hit[" << all << ":" << good << "]" << HGCScintillatorDetId(id);
            } else {
              edm::LogVerbatim("HitStudy") << "Hit[" << all << ":" << good << "]" << std::hex << id.rawId() << std::dec;
            }
          }
        } else {
          if (verbosity_ > 0) {
            if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
              edm::LogVerbatim("HitStudy")
                  << "Hit[" << all << ":" << good << "]" << HGCSiliconDetId(id) << " not valid *****";
            } else if (id.det() == DetId::HGCalHSc) {
              HGCScintillatorDetId hid1(id);
              HGCScintillatorDetId hid2(hid1.type(), hid1.layer(), hid1.ring(), hid1.iphi(), false, 0);
              bool ok = (std::find(validIds.begin(), validIds.end(), DetId(hid2)) != validIds.end());
              edm::LogVerbatim("HitStudy") << "Hit[" << all << ":" << good << "]" << hid1 << " not valid ***** but "
                                           << hid2 << " in list " << ok;
            } else {
              edm::LogVerbatim("HitStudy")
                  << "Hit[" << all << ":" << good << "]" << std::hex << id.rawId() << std::dec << " not valid *****";
            }
          }
        }
      }
    }
  }
  edm::LogVerbatim("HitStudy") << "Total hits = " << all << ":" << nhits << " Good DetIds = " << good
                               << " Invalid DetIds = " << (all - good);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGcalHitIdCheck);

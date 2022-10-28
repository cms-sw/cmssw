#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloSimUtils.h"

#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <fstream>
#include <string>
#include <vector>

class HGCalTestPartialWaferHits : public edm::one::EDAnalyzer<> {
public:
  HGCalTestPartialWaferHits(const edm::ParameterSet& ps);
  ~HGCalTestPartialWaferHits() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  const std::string g4Label_, caloHitSource_, nameSense_, missingFile_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  std::vector<int> wafers_;
  std::vector<int> dumpDets_;
};

HGCalTestPartialWaferHits::HGCalTestPartialWaferHits(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("moduleLabel")),
      caloHitSource_(ps.getParameter<std::string>("caloHitSource")),
      nameSense_(ps.getParameter<std::string>("nameSense")),
      missingFile_(ps.getParameter<std::string>("missingFile")),
      tok_calo_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, caloHitSource_))),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalSim") << "Test Hit ID using SimHits for " << nameSense_ << " with module Label: " << g4Label_
                               << "   Hits: " << caloHitSource_ << " Missing Wafer file " << missingFile_;
  if (!missingFile_.empty()) {
    edm::FileInPath filetmp("SimG4CMS/Calo/data/" + missingFile_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCalSim") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = CaloSimUtils::splitString(std::string(buffer));
        if (items.size() > 2) {
          int layer = std::atoi(items[0].c_str());
          int waferU = std::atoi(items[1].c_str());
          int waferV = std::atoi(items[2].c_str());
          wafers_.emplace_back(HGCalWaferIndex::waferIndex(layer, waferU, waferV, false));
        } else if (items.size() == 1) {
          int dumpdet = std::atoi(items[0].c_str());
          dumpDets_.emplace_back(dumpdet);
          edm::LogVerbatim("HGCalSim") << nameSense_ << " Dump detector " << dumpdet;
        }
      }
      edm::LogVerbatim("HGCalSim") << "HGCalTestPartialWaferHits::Reads in " << wafers_.size() << ":"
                                   << dumpDets_.size() << " wafer|detector information from " << fileName;
      fInput.close();
    }
  }
}

void HGCalTestPartialWaferHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("moduleLabel", "g4SimHits");
  desc.add<std::string>("caloHitSource", "HGCHitsEE");
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<std::string>("missingFile", "");
  descriptions.add("hgcalHitPartialEE", desc);
}

void HGCalTestPartialWaferHits::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  // get HGCalGeometry
  const HGCalGeometry* geom = &iS.getData(geomToken_);
  const HGCalDDDConstants& hgc = geom->topology().dddConstants();
  int firstLayer = hgc.getLayerOffset();
  // get the hit collection
  const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(tok_calo_);
  bool getHits = (hitsCalo.isValid());
  uint32_t nhits = (getHits) ? hitsCalo->size() : 0;
  uint32_t good(0), allSi(0), all(0);
  edm::LogVerbatim("HGCalSim") << "HGCalTestPartialWaferHits: Input flags Hits " << getHits << " with " << nhits
                               << " hits: Layer Offset " << firstLayer;

  if (getHits) {
    std::vector<PCaloHit> hits;
    hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
    if (!hits.empty()) {
      // Loop over all hits
      for (auto hit : hits) {
        ++all;
        DetId id(hit.id());
        if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
          ++allSi;
          HGCSiliconDetId hid(id);
          const auto& info = hgc.waferInfo(hid.layer(), hid.waferU(), hid.waferV());
          bool toCheck(false);
          if (!wafers_.empty()) {
            int indx = HGCalWaferIndex::waferIndex(firstLayer + hid.layer(), hid.waferU(), hid.waferV(), false);
            if (std::find(wafers_.begin(), wafers_.end(), indx) != wafers_.end())
              toCheck = true;
          } else if (!dumpDets_.empty()) {
            if ((std::find(dumpDets_.begin(), dumpDets_.end(), static_cast<int>(id.det())) != dumpDets_.end()) &&
                (info.part != HGCalTypes::WaferFull))
              toCheck = true;
          } else {
            // Only partial wafers
            toCheck = (info.part != HGCalTypes::WaferFull);
          }
          if (toCheck) {
            ++good;
            GlobalPoint pos = geom->getPosition(id);
            bool valid1 = geom->topology().valid(id);
            bool valid2 = hgc.isValidHex8(hid.layer(), hid.waferU(), hid.waferV(), hid.cellU(), hid.cellV());
            edm::LogVerbatim("HGCalSim") << "Hit[" << all << ":" << allSi << ":" << good << "]" << HGCSiliconDetId(id)
                                         << " Wafer Type:Part:Orient:Cassette " << info.type << ":" << info.part << ":"
                                         << info.orient << ":" << info.cassette << " at (" << pos.x() << ", " << pos.y()
                                         << ", " << pos.z() << ") Validity " << valid1 << ":" << valid2;
          }
        }
      }
    }
  }
  edm::LogVerbatim("HGCalSim") << "Total hits = " << all << ":" << nhits << " Good DetIds = " << allSi << ":" << good;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestPartialWaferHits);

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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/Calo/interface/CaloSimUtils.h"

#include <fstream>
#include <string>
#include <vector>

class HGCalTestPartialWaferRecHits : public edm::one::EDAnalyzer<> {
public:
  HGCalTestPartialWaferRecHits(const edm::ParameterSet& ps);
  ~HGCalTestPartialWaferRecHits() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  const std::string nameDetector_, missingFile_;
  const edm::InputTag source_;
  const edm::EDGetTokenT<HGCRecHitCollection> recHitSource_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcGeom_;
  std::vector<int> wafers_;
  std::vector<int> dumpDets_;
};

HGCalTestPartialWaferRecHits::HGCalTestPartialWaferRecHits(const edm::ParameterSet& ps)
    : nameDetector_(ps.getParameter<std::string>("detectorName")),
      missingFile_(ps.getParameter<std::string>("missingFile")),
      source_(ps.getParameter<edm::InputTag>("source")),
      recHitSource_(consumes<HGCRecHitCollection>(source_)),
      tok_hgcGeom_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameDetector_})) {
  edm::LogVerbatim("HGCalSim") << "Test Hit ID using RecHits for " << nameDetector_ << " with module Label: " << source_
                               << " Missing Wafer file " << missingFile_;
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
          edm::LogVerbatim("HGCalSim") << nameDetector_ << " Dump detector " << dumpdet;
        }
      }
      edm::LogVerbatim("HGCalSim") << "HGCalTestPartialWaferRecHits::Reads in " << wafers_.size() << ":"
                                   << dumpDets_.size() << " wafer|detector information from " << fileName;
      fInput.close();
    }
  }
}

void HGCalTestPartialWaferRecHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detectorName", "HGCalEESensitive");
  desc.add<std::string>("missingFile", "");
  desc.add<edm::InputTag>("source", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  descriptions.add("hgcalRecHitPartialEE", desc);
}

void HGCalTestPartialWaferRecHits::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  // get hgcalGeometry
  const HGCalGeometry* geom = &iS.getData(tok_hgcGeom_);
  const HGCalDDDConstants& hgc = geom->topology().dddConstants();
  int firstLayer = hgc.getLayerOffset();
  // get the hit collection
  const edm::Handle<HGCRecHitCollection>& theRecHitContainers = e.getHandle(recHitSource_);
  bool getHits = (theRecHitContainers.isValid());
  uint32_t nhits = (getHits) ? theRecHitContainers->size() : 0;
  uint32_t good(0), allSi(0), all(0);
  edm::LogVerbatim("HGCalSim") << "HGCalTestPartialWaferRecHits: Input flags Hits " << getHits << " with " << nhits
                               << " hits: Layer Offset " << firstLayer;

  if (getHits) {
    // Loop over all hits
    for (const auto& it : *(theRecHitContainers.product())) {
      ++all;
      DetId id(it.id());
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
          bool valid2 = hgc.isValidHex8(hid.layer(), hid.waferU(), hid.waferV(), hid.cellU(), hid.cellV(), false);
          edm::LogVerbatim("HGCalSim") << "Hit[" << all << ":" << allSi << ":" << good << "]" << hid
                                       << " Wafer Type:Part:Orient:Cassette " << info.type << ":" << info.part << ":"
                                       << info.orient << ":" << info.cassette << " at (" << pos.x() << ", " << pos.y()
                                       << ", " << pos.z() << ") Validity " << valid1 << ":" << valid2;
        }
      }
    }
  }
  edm::LogVerbatim("HGCalSim") << "Total hits = " << all << ":" << nhits << " Good DetIds = " << allSi << ":" << good;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestPartialWaferRecHits);

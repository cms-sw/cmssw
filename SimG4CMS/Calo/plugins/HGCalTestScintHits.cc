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

#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloSimUtils.h"

#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <fstream>
#include <string>
#include <vector>

class HGCalTestScintHits : public edm::one::EDAnalyzer<> {
public:
  HGCalTestScintHits(const edm::ParameterSet& ps);
  ~HGCalTestScintHits() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  const std::string g4Label_, caloHitSource_, nameSense_, tileFileName_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  std::vector<int> tiles_;
};

HGCalTestScintHits::HGCalTestScintHits(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("moduleLabel")),
      caloHitSource_(ps.getParameter<std::string>("caloHitSource")),
      nameSense_(ps.getParameter<std::string>("nameSense")),
      tileFileName_(ps.getParameter<std::string>("tileFileName")),
      tok_calo_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, caloHitSource_))),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalSim") << "Test Hit ID using SimHits for " << nameSense_ << " with module Label: " << g4Label_
                               << "   Hits: " << caloHitSource_ << " Tile file " << tileFileName_;
  if (!tileFileName_.empty()) {
    edm::FileInPath filetmp("SimG4CMS/Calo/data/" + tileFileName_);
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
          int ring = std::atoi(items[1].c_str());
          int phi = std::atoi(items[2].c_str());
          tiles_.emplace_back(HGCalTileIndex::tileIndex(layer, ring, phi));
        }
      }
      edm::LogVerbatim("HGCalSim") << "Reads in " << tiles_.size() << " tile information from " << fileName;
      fInput.close();
    }
  }
}

void HGCalTestScintHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("moduleLabel", "g4SimHits");
  desc.add<std::string>("caloHitSource", "HGCHitsHEback");
  desc.add<std::string>("nameSense", "HGCalHEScintillatorSensitive");
  desc.add<std::string>("tileFileName", "");
  descriptions.add("hgcalHitScintillator", desc);
}

void HGCalTestScintHits::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  // get hcalGeometry
  const HGCalGeometry* geom = &iS.getData(geomToken_);
  const HGCalDDDConstants& hgc = geom->topology().dddConstants();
  int firstLayer = hgc.firstLayer() - 1;
  // get the hit collection
  const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(tok_calo_);
  bool getHits = (hitsCalo.isValid());
  uint32_t nhits = (getHits) ? hitsCalo->size() : 0;
  uint32_t good(0), all(0);
  edm::LogVerbatim("HGCalSim") << "HGCalTestScintHits: Input flags Hits " << getHits << " with " << nhits
                               << " hits first Layer " << firstLayer;

  if (getHits) {
    std::vector<PCaloHit> hits;
    hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
    if (!hits.empty()) {
      // Loop over all hits
      for (auto hit : hits) {
        ++all;
        DetId id(hit.id());
        HGCScintillatorDetId hid(id);
        std::pair<int, int> typm = hgc.tileType(hid.layer(), hid.ring(), 0);
        if (typm.first >= 0) {
          hid.setType(typm.first);
          hid.setSiPM(typm.second);
          id = static_cast<DetId>(hid);
        }
        bool toCheck(true);
        if (!tiles_.empty()) {
          int indx = HGCalTileIndex::tileIndex(firstLayer + hid.layer(), hid.ring(), hid.iphi());
          if (std::find(tiles_.begin(), tiles_.end(), indx) != tiles_.end())
            toCheck = true;
        }
        if (toCheck) {
          ++good;
          GlobalPoint pos = geom->getPosition(id);
          bool valid1 = geom->topology().valid(id);
          bool valid2 = hgc.isValidTrap(hid.zside(), hid.layer(), hid.ring(), hid.iphi());
          edm::LogVerbatim("HGCalSim") << "Hit[" << all << ":" << good << "]" << hid << " at (" << pos.x() << ", "
                                       << pos.y() << ", " << pos.z() << ") Validity " << valid1 << ":" << valid2
                                       << " Energy " << hit.energy() << " Time " << hit.time();
        }
      }
    }
  }
  edm::LogVerbatim("HGCalSim") << "Total hits = " << all << ":" << nhits << " Good DetIds = " << good;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestScintHits);

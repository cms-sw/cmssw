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
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloSimUtils.h"
#include "SimG4CMS/Calo/interface/HGCGuardRing.h"

#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <fstream>
#include <sstream>
#include <string>
#include <map>

class HGCalTestGuardRing : public edm::one::EDAnalyzer<> {
public:
  HGCalTestGuardRing(const edm::ParameterSet& ps);
  ~HGCalTestGuardRing() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  const std::string nameSense_, waferFile_;
  const double guardRingOffset_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  std::map<HGCSiliconDetId, int> waferID_;
};

HGCalTestGuardRing::HGCalTestGuardRing(const edm::ParameterSet& ps)
    : nameSense_(ps.getParameter<std::string>("nameSense")),
      waferFile_(ps.getParameter<std::string>("waferFile")),
      guardRingOffset_(ps.getParameter<double>("guardRingOffset")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  DetId::Detector det = (nameSense_ != "HGCalHESiliconSensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
  edm::LogVerbatim("HGCalSim") << "Test Guard Ring Offset " << guardRingOffset_ << " for " << nameSense_ << ":" << det
                               << " for wafers read from file " << waferFile_;
  if (!waferFile_.empty()) {
    std::string thick[4] = {"h120", "l200", "l300", "h200"};
    int addType[4] = {HGCalTypes::WaferFineThin,
                      HGCalTypes::WaferCoarseThin,
                      HGCalTypes::WaferCoarseThick,
                      HGCalTypes::WaferFineThick};
    const int partTypeH[6] = {HGCalTypes::WaferFull,
                              HGCalTypes::WaferHalf2,
                              HGCalTypes::WaferChopTwoM,
                              HGCalTypes::WaferSemi2,
                              HGCalTypes::WaferSemi2,
                              HGCalTypes::WaferFive2};
    const int partTypeL[7] = {HGCalTypes::WaferFull,
                              HGCalTypes::WaferHalf,
                              HGCalTypes::WaferHalf,
                              HGCalTypes::WaferSemi,
                              HGCalTypes::WaferSemi,
                              HGCalTypes::WaferFive,
                              HGCalTypes::WaferThree};
    edm::FileInPath filetmp("SimG4CMS/Calo/data/" + waferFile_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCalSim") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = CaloSimUtils::splitString(std::string(buffer));
        if (items.size() > 6) {
          int layer = std::atoi(items[0].c_str());
          int waferU = std::atoi(items[4].c_str());
          int waferV = std::atoi(items[5].c_str());
          int thck = static_cast<int>(std::find(thick, thick + 4, items[2]) - thick);
          int type = (thck < 4) ? addType[thck] : 0;
          HGCSiliconDetId id(det, -1, type, layer, waferU, waferV, 0, 0);
          int orient = std::atoi(items[5].c_str());
          int part = std::atoi(items[1].c_str());
          if (part >= 0) {
            if (type == HGCalTypes::WaferFineThin)
              part = partTypeH[part];
            else
              part = partTypeL[part];
          }
          waferID_[id] = orient * 100 + part;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalSim") << "HGCalTestGuardRing::Reads " << id << " Orientatoin:Partial " >> orient >>
              ":" >> part;
#endif
        }
      }
      edm::LogVerbatim("HGCalSim") << "HGCalTestGuardRing::Reads in " << waferID_.size() << " wafer information from "
                                   << fileName;
      fInput.close();
    }
  }
}

void HGCalTestGuardRing::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<std::string>("waferFile", "testWafersEE.txt");
  desc.add<double>("guardRingOffset", 1.0);
  descriptions.add("hgcalTestGuardRingEE", desc);
}

void HGCalTestGuardRing::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  // get HGCalGeometry
  const HGCalGeometry* geom = &iS.getData(geomToken_);
  const HGCalDDDConstants& hgc = geom->topology().dddConstants();
  double waferSize = hgc.waferSize(false);
  HGCalCell wafer(waferSize, hgc.getUVMax(0), hgc.getUVMax(1));
  // get the hit collection
  edm::LogVerbatim("HGCalSim") << "HGCalTestGuardRing: Wafer Szie " << waferSize;

  // Loop over all IDs
  int all(0), allSi(0), good(0);
  for (std::map<HGCSiliconDetId, int>::const_iterator itr = waferID_.begin(); itr != waferID_.end(); ++itr) {
    HGCSiliconDetId id = itr->first;
    ++all;
    if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
      ++allSi;
      if (((id.det() == DetId::HGCalEE) && (nameSense_ == "HGCalEESensitive")) ||
          ((id.det() == DetId::HGCalHSi) && (nameSense_ == "HGCalHESiliconSensitive"))) {
        int partial = ((itr->second) % 100);
        int orient = (((itr->second) / 100) % 100);
        int type = id.type();
        int nCells = hgc.getUVMax(type);
        for (int u = 0; u < 2 * nCells; ++u) {
          for (int v = 0; v < 2 * nCells; ++v) {
            if (((v - u) < nCells) && ((u - v) <= nCells)) {
              HGCSiliconDetId hid(id.det(), id.zside(), id.type(), id.layer(), id.waferU(), id.waferV(), u, v);
              bool valid = (geom->topology()).valid(static_cast<DetId>(hid));
              if (valid) {
                ++good;
                int placeIndex = wafer.cellPlacementIndex(1, HGCalTypes::waferFrontBack(0), orient);
                std::pair<double, double> xy = wafer.cellUV2XY1(u, v, placeIndex, type);
                std::vector<std::pair<double, double> > wxy1 =
                    HGCalWaferMask::waferXY(partial, orient, -1, waferSize, 0.0, 0.0, 0.0);
                bool check1 = HGCGuardRing::insidePolygon(xy.first, xy.second, wxy1);
                std::ostringstream st1;
                for (unsigned int k1 = 0; k1 < wxy1.size(); ++k1)
                  st1 << " (" << wxy1[k1].first << ", " << wxy1[k1].second << ")";
                edm::LogVerbatim("HGCSim")
                    << "First " << hid << " Type:Partial:Orient:Place " << type << ":" << partial << ":" << orient
                    << ":" << placeIndex << " Boundary with " << wxy1.size() << " points: " << st1.str() << " check "
                    << check1 << " for (" << xy.first << ", " << xy.second << ")";

                std::vector<std::pair<double, double> > wxy2 =
                    HGCalWaferMask::waferXY(partial, orient, -1, waferSize, guardRingOffset_, 0.0, 0.0);
                bool check2 = HGCGuardRing::insidePolygon(xy.first, xy.second, wxy2);
                std::ostringstream st2;
                for (unsigned int k1 = 0; k1 < wxy2.size(); ++k1)
                  st2 << " (" << wxy2[k1].first << ", " << wxy2[k1].second << ")";
                edm::LogVerbatim("HGCSim") << "Second Offset " << guardRingOffset_ << " Boundary with " << wxy2.size()
                                           << " points: " << st2.str() << " check " << check2 << " for (" << xy.first
                                           << ", " << xy.second << ")";
              }
            }
          }
        }
      }
    }
  }
  edm::LogVerbatim("HGCalSim") << "Total hits = " << all << " Good Silicon DetIds = " << allSi << ":" << good;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestGuardRing);

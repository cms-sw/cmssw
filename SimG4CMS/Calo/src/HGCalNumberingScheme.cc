///////////////////////////////////////////////////////////////////////////////
// File: HGCalNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/CaloSimUtils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include <array>
#include <fstream>
#include <iostream>

//#define EDM_ML_DEBUG

HGCalNumberingScheme::HGCalNumberingScheme(const HGCalDDDConstants& hgc,
                                           const DetId::Detector& det,
                                           const std::string& name,
                                           const std::string& fileName)
    : hgcons_(hgc), mode_(hgc.geomMode()), det_(det), name_(name) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCalNumberingScheme for " << name_ << " Det " << det_ << " Mode " << mode_
                             << ":" << HGCalGeometryMode::Hexagon8Full << ":" << HGCalGeometryMode::Hexagon8 << ":"
                             << HGCalGeometryMode::Hexagon8File << ":" << HGCalGeometryMode::Hexagon8Module << ":"
                             << ":" << HGCalGeometryMode::Hexagon8Cassette << ":" << HGCalGeometryMode::Trapezoid << ":"
                             << HGCalGeometryMode::TrapezoidFile << ":" << HGCalGeometryMode::TrapezoidModule << ":"
                             << HGCalGeometryMode::TrapezoidCassette;
#endif
  firstLayer_ = hgcons_.getLayerOffset();
  if (!fileName.empty()) {
    edm::FileInPath filetmp1("SimG4CMS/Calo/data/" + fileName);
    std::string filetmp2 = filetmp1.fullPath();
    std::ifstream fInput(filetmp2.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCalSim") << "Cannot open file " << filetmp2;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = CaloSimUtils::splitString(std::string(buffer));
        if (items.size() == 3) {
          if (hgcons_.waferHexagon8File()) {
            int layer = std::atoi(items[0].c_str());
            int waferU = std::atoi(items[1].c_str());
            int waferV = std::atoi(items[2].c_str());
            indices_.emplace_back(HGCalWaferIndex::waferIndex(layer, waferU, waferV, false));
          } else if (hgcons_.tileTrapezoid()) {
            int layer = std::atoi(items[0].c_str());
            int ring = std::atoi(items[1].c_str());
            int iphi = std::atoi(items[2].c_str());
            indices_.emplace_back(HGCalTileIndex::tileIndex(layer, ring, iphi));
          }
        } else if (items.size() == 1) {
          int dumpdet = std::atoi(items[0].c_str());
          dumpDets_.emplace_back(dumpdet);
        } else if (items.size() == 4) {
          int idet = std::atoi(items[0].c_str());
          int layer = std::atoi(items[1].c_str());
          int zside = std::atoi(items[2].c_str());
          int cassette = std::atoi(items[3].c_str());
          dumpCassette_.emplace_back(HGCalCassette::cassetteIndex(idet, layer, zside, cassette));
        }
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalSim") << "Reads in " << indices_.size() << ":" << dumpDets_.size() << ":"
                                   << dumpCassette_.size() << " component information from " << filetmp2
                                   << " Layer Offset " << firstLayer_;
#endif
      fInput.close();
    }
  }
}

HGCalNumberingScheme::~HGCalNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Deleting HGCalNumberingScheme";
#endif
}

uint32_t HGCalNumberingScheme::getUnitID(int layer, int module, int cell, int iz, const G4ThreeVector& pos, double& wt) {
  // module is the copy number of the wafer as placed in the layer
  uint32_t index(0);
  wt = 1.0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme:: input Layer " << layer << " Module " << module << " Cell "
                             << cell << " iz " << iz << " Position " << pos;
#endif
  if (hgcons_.waferHexagon8()) {
    int cellU(0), cellV(0), waferType(-1), waferU(0), waferV(0);
    if (cell >= 0) {
      waferType = HGCalTypes::getUnpackedType(module);
      waferU = HGCalTypes::getUnpackedU(module);
      waferV = HGCalTypes::getUnpackedV(module);
      cellU = HGCalTypes::getUnpackedCellU(cell);
      cellV = HGCalTypes::getUnpackedCellV(cell);
    } else if (mode_ != HGCalGeometryMode::Hexagon8) {
      int zside = (pos.z() > 0) ? 1 : -1;
      double xx = zside * pos.x();
      int wU = HGCalTypes::getUnpackedU(module);
      int wV = HGCalTypes::getUnpackedV(module);
      bool debug(false);
      if (!indices_.empty()) {
        int indx = HGCalWaferIndex::waferIndex(firstLayer_ + layer, wU, wV, false);
        if (std::find(indices_.begin(), indices_.end(), indx) != indices_.end())
          debug = true;
      }
      if (!dumpDets_.empty()) {
        if ((std::find(dumpDets_.begin(), dumpDets_.end(), det_) != dumpDets_.end()) &&
            (hgcons_.waferInfo(layer, wU, wV).part != HGCalTypes::WaferFull))
          debug = true;
      }
      if (!dumpCassette_.empty()) {
        int indx = HGCalWaferIndex::waferIndex(firstLayer_ + layer, wU, wV, false);
        auto ktr = hgcons_.getParameter()->waferInfoMap_.find(indx);
        if (ktr != hgcons_.getParameter()->waferInfoMap_.end()) {
          if (std::find(dumpCassette_.begin(),
                        dumpCassette_.end(),
                        HGCalCassette::cassetteIndex(det_, firstLayer_ + layer, zside, (ktr->second).cassette)) !=
              dumpCassette_.end())
            debug = true;
        }
      }
      hgcons_.waferFromPosition(xx, pos.y(), zside, layer, waferU, waferV, cellU, cellV, waferType, wt, false, debug);
    }
    if (waferType >= 0) {
      if (hgcons_.waferHexagon8File()) {
        int type = hgcons_.waferType(layer, waferU, waferV, true);
        if (type != waferType) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme:: " << name_ << " Layer|u|v|Index|module|cell " << layer
                                     << ":" << waferU << ":" << waferV << ":"
                                     << HGCalWaferIndex::waferIndex(layer, waferU, waferV, false) << ":" << module
                                     << ":" << cell << " has a type mismatch " << waferType << ":" << type;
#endif
          if (type != HGCSiliconDetId::HGCalCoarseThick)
            waferType = type;
        }
      }
      index = HGCSiliconDetId(det_, iz, waferType, layer, waferU, waferV, cellU, cellV).rawId();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "OK WaferType " << waferType << " Wafer " << waferU << ":" << waferV << " Cell "
                                 << cellU << ":" << cellV << " input " << cell << " wt " << wt << " Mode " << mode_;
    } else {
      edm::LogVerbatim("HGCSim") << "Bad WaferType " << waferType << " for Layer:u:v " << layer << ":" << waferU << ":"
                                 << waferV;
#endif
    }
  } else if (hgcons_.tileTrapezoid()) {
    std::array<int, 3> id = hgcons_.assignCellTrap(pos.x(), pos.y(), pos.z(), layer, false);
    if (id[2] >= 0) {
      std::pair<int, int> typm = hgcons_.tileType(layer, id[0], 0);
      HGCScintillatorDetId detId(id[2], layer, iz * id[0], id[1], false, 0);
      if (typm.first >= 0) {
        detId.setType(typm.first);
        detId.setSiPM(typm.second);
      }
      index = detId.rawId();
      bool debug(false);
      if (!indices_.empty()) {
        int indx = HGCalTileIndex::tileIndex(layer, id[0], id[1]);
        if (std::find(indices_.begin(), indices_.end(), indx) != indices_.end())
          debug = true;
      }
      if (debug)
        edm::LogVerbatim("HGCSim") << "Radius/Phi " << id[0] << ":" << id[1] << " Type " << id[2] << ":" << typm.first
                                   << " SiPM " << typm.second << ":" << hgcons_.tileSiPM(typm.second) << " Layer "
                                   << layer << " z " << iz << " " << detId << " wt " << wt << " position " << pos
                                   << " R " << pos.perp();
#ifdef EDM_ML_DEBUG
    } else {
      edm::LogVerbatim("HGCSim") << "Radius/Phi " << id[0] << ":" << id[1] << " Type " << id[2] << " Layer|iz " << layer
                                 << ":" << iz << " ERROR";
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  bool matchOnly = ((mode_ == HGCalGeometryMode::Hexagon8Module) || (mode_ == HGCalGeometryMode::Hexagon8Cassette));
  bool debug = hgcons_.waferHexagon8File();
  if (debug)
    edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme::i/p " << det_ << ":" << layer << ":" << module << ":" << cell
                               << ":" << iz << ":" << pos.x() << ":" << pos.y() << ":" << pos.z() << " ID " << std::hex
                               << index << std::dec << " wt " << wt;
  checkPosition(index, pos, matchOnly, debug);
#endif
  return index;
}

void HGCalNumberingScheme::checkPosition(uint32_t index, const G4ThreeVector& pos, bool matchOnly, bool debug) const {
  std::pair<float, float> xy;
  bool ok(false);
  double z1(0), tolR(14.0), tolZ(1.0);
  int lay(-1);
  if (index == 0) {
  } else if (DetId(index).det() == DetId::HGCalHSi) {
    HGCSiliconDetId id = HGCSiliconDetId(index);
    lay = id.layer();
    xy = hgcons_.locateCell(
        id.zside(), lay, id.waferU(), id.waferV(), id.cellU(), id.cellV(), false, true, false, false);
    z1 = hgcons_.waferZ(lay, false);
    ok = true;
    tolR = 14.0;
    tolZ = 1.0;
  } else if (DetId(index).det() == DetId::HGCalHSc) {
    HGCScintillatorDetId id = HGCScintillatorDetId(index);
    lay = id.layer();
    xy = hgcons_.locateCellTrap(id.zside(), lay, id.ietaAbs(), id.iphi(), false, false);
    z1 = hgcons_.waferZ(lay, false);
    ok = true;
    tolR = 50.0;
    tolZ = 5.0;
  }
  if (ok) {
    double r1 = std::sqrt(xy.first * xy.first + xy.second * xy.second);
    double r2 = pos.perp();
    double z2 = std::abs(pos.z());
    std::pair<double, double> zrange = hgcons_.rangeZ(false);
    std::pair<double, double> rrange = hgcons_.rangeR(z2, false);
    bool match = (std::abs(r1 - r2) < tolR) && (std::abs(z1 - z2) < tolZ);
    bool inok = ((r2 >= rrange.first) && (r2 <= rrange.second) && (z2 >= zrange.first) && (z2 <= zrange.second));
    bool outok = ((r1 >= rrange.first) && (r1 <= rrange.second) && (z1 >= zrange.first) && (z1 <= zrange.second));
    std::string ck = (((r1 < rrange.first - tolR) || (r1 > rrange.second + tolR) || (z1 < zrange.first - tolZ) ||
                       (z1 > zrange.second + tolZ))
                          ? "***** ERROR *****"
                          : "");
    if (matchOnly && match)
      ck = "";
    if (!(match && inok && outok) || debug) {
      edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme::Detector " << det_ << " Layer " << lay << " R " << r2 << ":"
                                 << r1 << ":" << rrange.first << ":" << rrange.second << " Z " << z2 << ":" << z1 << ":"
                                 << zrange.first << ":" << zrange.second << " Match " << match << ":" << inok << ":"
                                 << outok << " " << ck;
      edm::LogVerbatim("HGCSim") << "Original " << pos.x() << ":" << pos.y() << " return " << xy.first << ":"
                                 << xy.second;
      if ((DetId(index).det() == DetId::HGCalEE) || (DetId(index).det() == DetId::HGCalHSi)) {
        int zside = (pos.z() > 0) ? 1 : -1;
        double wt(0), xx(zside * pos.x());
        int waferU, waferV, cellU, cellV, waferType;
        hgcons_.waferFromPosition(xx, pos.y(), zside, lay, waferU, waferV, cellU, cellV, waferType, wt, false, true);
        xy = hgcons_.locateCell(zside, lay, waferU, waferV, cellU, cellV, false, true, false, true);
        double dx = (xx - xy.first);
        double dy = (pos.y() - xy.second);
        double dR = std::sqrt(dx * dx + dy * dy);
        ck = (dR > tolR) ? " ***** ERROR *****" : "";
        edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme " << HGCSiliconDetId(index) << " original position " << xx
                                   << ":" << pos.y() << " derived " << xy.first << ":" << xy.second << " Difference "
                                   << dR << ck;
      }
    }
  }
}

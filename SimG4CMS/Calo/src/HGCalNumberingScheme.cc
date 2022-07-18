///////////////////////////////////////////////////////////////////////////////
// File: HGCalNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include <array>
#include <iostream>

//#define EDM_ML_DEBUG

HGCalNumberingScheme::HGCalNumberingScheme(const HGCalDDDConstants& hgc,
                                           const DetId::Detector& det,
                                           const std::string& name)
    : hgcons_(hgc), mode_(hgc.geomMode()), det_(det), name_(name) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCalNumberingScheme for " << name_ << " Det " << det_ << " Mode " << mode_
                             << ":" << HGCalGeometryMode::Hexagon8Full << ":" << HGCalGeometryMode::Hexagon8 << ":"
                             << HGCalGeometryMode::Hexagon8File << ":" << HGCalGeometryMode::Hexagon8Module << ":"
                             << ":" << HGCalGeometryMode::Hexagon8Cassette << ":" << HGCalGeometryMode::Trapezoid << ":"
                             << HGCalGeometryMode::TrapezoidFile << ":" << HGCalGeometryMode::TrapezoidModule << ":"
                             << HGCalGeometryMode::TrapezoidCassette;
#endif
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
      double xx = (pos.z() > 0) ? pos.x() : -pos.x();
      hgcons_.waferFromPosition(xx, pos.y(), layer, waferU, waferV, cellU, cellV, waferType, wt, false, false);
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
#ifdef EDM_ML_DEBUG
      int lay = layer + hgcons_.getLayerOffset();
      edm::LogVerbatim("HGCSim") << "Radius/Phi " << id[0] << ":" << id[1] << " Type " << id[2] << ":" << typm.first
                                 << " SiPM " << typm.second << ":" << hgcons_.tileSiPM(typm.second) << " Layer "
                                 << layer << ":" << lay << " z " << iz << " " << detId;
    } else {
      edm::LogVerbatim("HGCSim") << "Radius/Phi " << id[0] << ":" << id[1] << " Type " << id[2] << " Layer|iz " << layer
                                 << ":" << iz << " ERROR";
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  bool matchOnly = (mode_ == HGCalGeometryMode::Hexagon8Module);
  bool debug = (mode_ == HGCalGeometryMode::Hexagon8Module);
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
    xy = hgcons_.locateCell(lay, id.waferU(), id.waferV(), id.cellU(), id.cellV(), false, true);
    z1 = hgcons_.waferZ(lay, false);
    ok = true;
    tolR = 14.0;
    tolZ = 1.0;
  } else if (DetId(index).det() == DetId::HGCalHSc) {
    HGCScintillatorDetId id = HGCScintillatorDetId(index);
    lay = id.layer();
    xy = hgcons_.locateCellTrap(lay, id.ietaAbs(), id.iphi(), false);
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
        double wt = 0, xx = ((pos.z() > 0) ? pos.x() : -pos.x());
        int waferU, waferV, cellU, cellV, waferType;
        hgcons_.waferFromPosition(xx, pos.y(), lay, waferU, waferV, cellU, cellV, waferType, wt, false, false);
        xy = hgcons_.locateCell(lay, waferU, waferV, cellU, cellV, false, true, true);
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

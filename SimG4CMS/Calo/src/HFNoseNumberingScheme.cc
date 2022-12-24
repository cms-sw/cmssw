///////////////////////////////////////////////////////////////////////////////
// File: HFNoseNumberingScheme.cc
// Description: Numbering scheme for HFNose detector
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HFNoseNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include <iostream>

//#define EDM_ML_DEBUG

HFNoseNumberingScheme::HFNoseNumberingScheme(const HGCalDDDConstants& hgc) : hgcons_(hgc), mode_(hgc.geomMode()) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HFNoseNumberingScheme";
#endif
}

uint32_t HFNoseNumberingScheme::getUnitID(
    int layer, int module, int cell, int iz, const G4ThreeVector& pos, double& wt) {
  // module is the copy number of the wafer as placed in the layer
  uint32_t index(0);
  wt = 1.0;
  int cellU(0), cellV(0), waferType(-1), waferU(0), waferV(0);
  if (cell >= 0) {
    waferType = HGCalTypes::getUnpackedType(module);
    waferU = HGCalTypes::getUnpackedU(module);
    waferV = HGCalTypes::getUnpackedV(module);
    cellU = HGCalTypes::getUnpackedCellU(cell);
    cellV = HGCalTypes::getUnpackedCellV(cell);
  } else if (mode_ == HGCalGeometryMode::Hexagon8Full) {
    int zside = (pos.z() > 0) ? 1 : -1;
    double xx = zside * pos.x();
    hgcons_.waferFromPosition(xx, pos.y(), zside, layer, waferU, waferV, cellU, cellV, waferType, wt, false, false);
  }
  if (waferType >= 0) {
    index = HFNoseDetId(iz, waferType, layer, waferU, waferV, cellU, cellV).rawId();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HFNSim") << "OK WaferType " << waferType << " Wafer " << waferU << ":" << waferV << " Cell "
                               << cellU << ":" << cellV;
  } else {
    edm::LogVerbatim("HFNSim") << "Bad WaferType " << waferType;
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFNSim") << "HFNoseNumberingScheme::i/p " << layer << ":" << module << ":" << cell << ":" << iz
                             << ":" << pos.x() << ":" << pos.y() << ":" << pos.z() << " ID " << std::hex << index
                             << std::dec << " wt " << wt;
  checkPosition(index, pos);
#endif
  return index;
}

void HFNoseNumberingScheme::checkPosition(uint32_t index, const G4ThreeVector& pos) const {
  std::pair<float, float> xy;
  bool ok(false);
  double z1(0), tolR(10.0), tolZ(1.0);
  int lay(-1);
  if (index == 0) {
  } else if ((DetId(index).det() == DetId::Forward) && (DetId(index).subdetId() == static_cast<int>(HFNose))) {
    HFNoseDetId id = HFNoseDetId(index);
    lay = id.layer();
    xy = hgcons_.locateCell(
        id.zside(), lay, id.waferU(), id.waferV(), id.cellU(), id.cellV(), false, true, false, false);
    z1 = hgcons_.waferZ(lay, false);
    ok = true;
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
    if (!(match && inok && outok)) {
      edm::LogVerbatim("HGCSim") << "HFNoseNumberingScheme::Detector " << DetId(index).det() << " Layer " << lay
                                 << " R " << r2 << ":" << r1 << ":" << rrange.first << ":" << rrange.second << " Z "
                                 << z2 << ":" << z1 << ":" << zrange.first << ":" << zrange.second << " Match " << match
                                 << ":" << inok << ":" << outok << " " << ck;
      edm::LogVerbatim("HGCSim") << "Original " << pos.x() << ":" << pos.y() << " return " << xy.first << ":"
                                 << xy.second;
      if ((DetId(index).det() == DetId::Forward) && (DetId(index).subdetId() == static_cast<int>(HFNose))) {
        int zside = (pos.z() > 0) ? 1 : -1;
        double wt = 0, xx = (zside * pos.x());
        int waferU, waferV, cellU, cellV, waferType;
        hgcons_.waferFromPosition(xx, pos.y(), zside, lay, waferU, waferV, cellU, cellV, waferType, wt, false, true);
        xy = hgcons_.locateCell(zside, lay, waferU, waferV, cellU, cellV, false, true, false, true);
        edm::LogVerbatim("HGCSim") << "HFNoseNumberingScheme " << HFNoseDetId(index) << " position " << xy.first << ":"
                                   << xy.second;
      }
    }
  }
}

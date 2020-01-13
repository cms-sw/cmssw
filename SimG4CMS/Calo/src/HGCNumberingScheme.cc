///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "DataFormats/Math/interface/FastMath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>

//#define EDM_ML_DEBUG

HGCNumberingScheme::HGCNumberingScheme(const HGCalDDDConstants& hgc, std::string& name) : hgcons_(hgc) {
  edm::LogVerbatim("HGCSim") << "Creating HGCNumberingScheme for " << name;
}

HGCNumberingScheme::~HGCNumberingScheme() { edm::LogVerbatim("HGCSim") << "Deleting HGCNumberingScheme"; }

uint32_t HGCNumberingScheme::getUnitID(
    ForwardSubdetector subdet, int layer, int module, int cell, int iz, const G4ThreeVector& pos) {
  // module is the copy number of the wafer as placed in the layer
  int icell(0), celltyp(0), wafer(0);
  uint32_t index(0);
  if (hgcons_.geomMode() == HGCalGeometryMode::HexagonFull) {
    if (cell >= 0) {
      wafer = hgcons_.waferFromCopy(module);
      celltyp = cell / 1000;
      icell = cell % 1000;
    } else {
      hgcons_.waferFromPosition(pos.x(), pos.y(), wafer, icell, celltyp);
    }
    if (wafer >= 0) {
      if (celltyp != 1)
        celltyp = 0;
      index = HGCalTestNumbering::packHexagonIndex((int)subdet, iz, layer, wafer, celltyp, icell);
    }
  } else if (hgcons_.geomMode() == HGCalGeometryMode::Hexagon) {
    wafer = hgcons_.waferFromCopy(module);
    celltyp = cell / 1000;
    icell = cell % 1000;
    if (celltyp != 1)
      celltyp = 0;

    index = HGCalTestNumbering::packHexagonIndex((int)subdet, iz, layer, wafer, celltyp, icell);
    //check if it fits
    if (!hgcons_.isValidHex(layer, wafer, icell, false)) {
      index = 0;
      edm::LogError("HGCSim") << "[HGCNumberingScheme] ID out of bounds :"
                              << " Subdet= " << subdet << " Zside= " << iz << " Layer= " << layer << " Wafer= " << wafer
                              << ":" << module << " CellType= " << celltyp << " Cell= " << icell;
    }
  }
#ifdef EDM_ML_DEBUG
  int subd, zside, lay, sector, subsector, cellx;
  HGCalTestNumbering::unpackHexagonIndex(index, subd, zside, lay, sector, subsector, cellx);
  edm::LogVerbatim("HGCSim") << "HGCNumberingScheme::i/p " << subdet << ":" << layer << ":" << module << ":" << iz
                             << ":" << wafer << ":" << celltyp << ":" << icell << ":" << std::hex << index << std::dec
                             << " Output " << subd << ":" << lay << ":" << zside << ":" << sector << ":" << subsector
                             << ":" << cellx;
#endif
  return index;
}

int HGCNumberingScheme::assignCell(float x, float y, int layer) {
  std::pair<int, int> phicell = hgcons_.assignCell(x, y, layer, 0, false);
  return phicell.second;
}

std::pair<float, float> HGCNumberingScheme::getLocalCoords(int cell, int layer) {
  return hgcons_.locateCell(cell, layer, 0, false);
}

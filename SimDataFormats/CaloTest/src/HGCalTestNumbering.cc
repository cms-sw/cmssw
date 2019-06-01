#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include <iostream>

//#define EDM_ML_DEBUG

uint32_t HGCalTestNumbering::packSquareIndex(int zp, int lay, int sec, int subsec, int cell) {
  if (!HGCalTestNumbering::isValidSquare(zp, lay, sec, subsec, lay)) {
    zp = lay = sec = subsec = cell = 0;
  }

  uint32_t rawid = 0;
  rawid |= ((cell & kHGCalCellSMask) << kHGCalCellSOffset);
  rawid |= ((sec & kHGCalSectorSMask) << kHGCalSectorSOffset);
  if (subsec < 0)
    subsec = 0;
  rawid |= ((subsec & kHGCalSubSectorSMask) << kHGCalSubSectorSOffset);
  rawid |= ((lay & kHGCalLayerSMask) << kHGCalLayerSOffset);
  if (zp > 0)
    rawid |= ((zp & kHGCalZsideSMask) << kHGCalZsideSOffset);
  return rawid;
}

uint32_t HGCalTestNumbering::packHexagonIndex(int subdet, int zp, int lay, int wafer, int celltyp, int cell) {
  if (!HGCalTestNumbering::isValidHexagon(subdet, zp, lay, wafer, celltyp, cell)) {
    subdet = zp = lay = wafer = celltyp = cell = 0;
  }

  uint32_t rawid = 0;
  rawid |= ((cell & kHGCalCellHMask) << kHGCalCellHOffset);
  rawid |= ((celltyp & kHGCalCellTypHMask) << kHGCalCellTypHOffset);
  rawid |= ((wafer & kHGCalWaferHMask) << kHGCalWaferHOffset);
  rawid |= ((lay & kHGCalLayerHMask) << kHGCalLayerHOffset);
  if (zp > 0)
    rawid |= ((zp & kHGCalZsideHMask) << kHGCalZsideHOffset);
  rawid |= ((subdet & kHGCalSubdetHMask) << kHGCalSubdetHOffset);
  return rawid;
}

void HGCalTestNumbering::unpackSquareIndex(const uint32_t& idx, int& zp, int& lay, int& sec, int& subsec, int& cell) {
  cell = (idx >> kHGCalCellSOffset) & kHGCalCellSMask;
  subsec = ((idx >> kHGCalSubSectorSOffset) & kHGCalSubSectorSMask ? 1 : -1);
  sec = (idx >> kHGCalSectorSOffset) & kHGCalSectorSMask;
  lay = (idx >> kHGCalLayerSOffset) & kHGCalLayerSMask;
  zp = ((idx >> kHGCalZsideSOffset) & kHGCalZsideSMask ? 1 : -1);
}

void HGCalTestNumbering::unpackHexagonIndex(
    const uint32_t& idx, int& subdet, int& zp, int& lay, int& wafer, int& celltyp, int& cell) {
  cell = (idx >> kHGCalCellHOffset) & kHGCalCellHMask;
  celltyp = (idx >> kHGCalCellTypHOffset) & kHGCalCellTypHMask;
  wafer = (idx >> kHGCalWaferHOffset) & kHGCalWaferHMask;
  lay = (idx >> kHGCalLayerHOffset) & kHGCalLayerHMask;
  zp = ((idx >> kHGCalZsideHOffset) & kHGCalZsideHMask ? 1 : -1);
  subdet = (idx >> kHGCalSubdetHOffset) & kHGCalSubdetHMask;
}

bool HGCalTestNumbering::isValidSquare(int zp, int lay, int sec, int subsec, int cell) {
  if (cell > kHGCalCellSMask || sec > kHGCalSectorSMask || subsec > kHGCalSubSectorSMask || lay > kHGCalLayerSMask) {
#ifdef EDM_ML_DEBUG
    std::cout << "[HGCalTestNumbering] request for new id for layer=" << lay << " zp=" << zp << " sector=" << sec
              << " subsec=" << subsec << " cell=" << cell << " has one or more fields out of bounds and will be reset"
              << std::endl;
#endif
    return false;
  }
  return true;
}

bool HGCalTestNumbering::isValidHexagon(int subdet, int zp, int lay, int wafer, int celltyp, int cell) {
  if (cell > kHGCalCellHMask || celltyp > kHGCalCellTypHMask || wafer > kHGCalWaferHMask || lay > kHGCalLayerSMask ||
      subdet > kHGCalSubdetHMask) {
#ifdef EDM_ML_DEBUG
    std::cout << "[HGCalTestNumbering] request for new id for layer=" << lay << " zp=" << zp << " wafer=" << wafer
              << " celltyp=" << celltyp << " cell=" << cell << " for subdet=" << subdet
              << " has one or more fields out of bounds and will be reset" << std::endl;
#endif
    return false;
  }
  return true;
}

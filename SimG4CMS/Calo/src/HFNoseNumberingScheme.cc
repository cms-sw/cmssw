///////////////////////////////////////////////////////////////////////////////
// File: HFNoseNumberingScheme.cc
// Description: Numbering scheme for HFNose detector
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HFNoseNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
    waferType = module / 1000000;
    waferU = module % 100;
    if ((module / 10000) % 10 > 0)
      waferU = -waferU;
    waferV = (module / 100) % 100;
    if ((module / 100000) % 10 > 0)
      waferV = -waferV;
    cellU = cell % 100;
    cellV = (cell / 100) % 100;
  } else if (mode_ == HGCalGeometryMode::Hexagon8Full) {
    double xx = (pos.z() > 0) ? pos.x() : -pos.x();
    hgcons_.waferFromPosition(xx, pos.y(), layer, waferU, waferV, cellU, cellV, waferType, wt);
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
#endif
  return index;
}

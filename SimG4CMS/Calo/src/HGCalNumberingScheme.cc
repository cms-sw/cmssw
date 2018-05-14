///////////////////////////////////////////////////////////////////////////////
// File: HGCalNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include <iostream>

//#define EDM_ML_DEBUG

HGCalNumberingScheme::HGCalNumberingScheme(const HGCalDDDConstants& hgc, 
					   const DetId::Detector& det,
					   const std::string & name) :
  hgcons_(hgc), mode_(hgc.geomMode()), det_(det), name_(name) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Creating HGCalNumberingScheme for " << name_
			     << " Det " << det_;
#endif
}

HGCalNumberingScheme::~HGCalNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Deleting HGCalNumberingScheme";
#endif
}

uint32_t HGCalNumberingScheme::getUnitID(int layer, int module, int cell,
					 int iz, const G4ThreeVector &pos,
					 double& wt) {
  // module is the copy number of the wafer as placed in the layer
  uint32_t index(0);
  wt = 1.0;
  if ((mode_ == HGCalGeometryMode::Hexagon8Full) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int      cellU(0), cellV(0), waferType(-1), waferU(0), waferV(0);
    if (cell >= 0) {
      waferType = module/1000000;
      waferU    = module%100;
      if ((module/10000)%10  > 0) waferU = -waferU;
      waferV    = (module/100)%100;
      if ((module/100000)%10 > 0) waferV = -waferV;
      cellU     = cell%100;
      cellV     = (cell/100)%100;
    } else if (mode_ == HGCalGeometryMode::Hexagon8Full) {
      double xx = (pos.z() > 0) ? pos.x() : -pos.x();
      hgcons_.waferFromPosition(xx,pos.y(),layer,waferU,waferV,cellU,
				cellV,waferType,wt);
    }
    if (waferType >= 0) {
      index   = HGCSiliconDetId(det_,iz,waferType,layer,waferU,waferV,cellU,cellV).rawId();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "WaferType " << waferType << " Wafer "
				 << waferU << ":" << waferV << " Cell "
				 << cellU << ":" << cellV;
#endif
    }
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    std::array<int,3> id = hgcons_.assignCellTrap(pos.x(),pos.y(),pos.z(),
						  layer,false);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Eta/Phi " << id[0] << ":" << id[1]
			       << " Type " << id[2] << " Layer|iz " << layer
			       << ":" << iz;
#endif
    if (id[2] >= 0) 
      index   = HGCScintillatorDetId(id[2], layer, iz*id[0], id[1]).rawId();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme::i/p " << det_ << ":" 
			     << layer << ":" << module << ":" << cell << ":" 
			     << iz << ":" << pos.x() << ":" << pos.y() << ":"
			     << pos.z() << " ID " << std::hex << index 
			     << std::dec << " wt " << wt;
#endif
  return index;
}

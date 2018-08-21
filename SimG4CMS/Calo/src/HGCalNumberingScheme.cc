///////////////////////////////////////////////////////////////////////////////
// File: HGCalNumberingScheme.cc
// Description: Numbering scheme for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
      (mode_ == HGCalGeometryMode::Hexagon8)) {
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
      index   = HGCSiliconDetId(det_,iz,waferType,layer,waferU,waferV,
				cellU,cellV).rawId();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "OK WaferType " << waferType << " Wafer "
				 << waferU << ":" << waferV << " Cell "
				 << cellU << ":" << cellV;
    } else {
      edm::LogVerbatim("HGCSim") << "Bad WaferType " << waferType;
#endif
    }
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    std::array<int,3> id = hgcons_.assignCellTrap(pos.x(),pos.y(),pos.z(),
						  layer,false);
    if (id[2] >= 0) {
      index   = HGCScintillatorDetId(id[2], layer, iz*id[0], id[1]).rawId();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "Eta/Phi " << id[0] << ":" << id[1]
				 << " Type " << id[2] << " Layer|iz " 
				 << layer << ":" << iz << " " 
				 << HGCScintillatorDetId(index);
    } else {
      edm::LogVerbatim("HGCSim") << "Eta/Phi " << id[0] << ":" << id[1]
				 << " Type " << id[2] << " Layer|iz " << layer
				 << ":" << iz;
#endif
    }

  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme::i/p " << det_ << ":" 
			     << layer << ":" << module << ":" << cell << ":" 
			     << iz << ":" << pos.x() << ":" << pos.y() << ":"
			     << pos.z() << " ID " << std::hex << index 
			     << std::dec << " wt " << wt;
  checkPosition(index,pos);
#endif
  return index;
}

void HGCalNumberingScheme::checkPosition(uint32_t index,
					 const G4ThreeVector &pos) const {

  std::pair<float,float> xy;
  bool                   ok(false);
  double                 z1(0), drMax(10.0), dzMax(1.0);
  int                    lay(-1);
  if (index == 0) {
  } else if (DetId(index).det() == DetId::HGCalHSi) {
    HGCSiliconDetId id = HGCSiliconDetId(index);
    lay= id.layer();
    xy = hgcons_.locateCell(lay,id.waferU(),id.waferV(),id.cellU(),
			    id.cellV(),false,true);
    z1 = hgcons_.waferZ(lay,false);
    ok = true;
    drMax = 10.0; dzMax = 1.0;
  } else if (DetId(index).det() == DetId::HGCalHSc) {
    HGCScintillatorDetId id = HGCScintillatorDetId(index);
    lay= id.layer();
    xy = hgcons_.locateCellTrap(lay,id.ietaAbs(),id.iphi(),false);
    z1 = hgcons_.waferZ(lay,false);
    ok = true;
    drMax = 50.0; dzMax = 5.0;
  }
  if (ok) {
    double r1 = std::sqrt(xy.first*xy.first+xy.second*xy.second);
    double r2 = pos.perp();
    double z2 = std::abs(pos.z());
    std::pair<double,double> zrange = hgcons_.rangeZ(false);
    std::pair<double,double> rrange = hgcons_.rangeR(z2,false);
    bool match= (std::abs(r1-r2) < drMax) && (std::abs(z1-z2) < dzMax);
    bool inok = ((r2 >= rrange.first) && (r2 <= rrange.second) &&
		 (z2 >= zrange.first) && (z2 <= zrange.second));
    bool outok= ((r1 >= rrange.first) && (r1 <= rrange.second) &&
		 (z1 >= zrange.first) && (z1 <= zrange.second));
    std::string ck = (((r1 < rrange.first-10.0) || (r1 > rrange.second+10.0) ||
		       (z1 < zrange.first-5.0) || (z1 > zrange.second+5.0)) ? 
		      "***** ERROR *****" : "");
    if (!(match && inok && outok)) {
      edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme::Detector " << det_ 
				 << " Layer " << lay << " R " << r2 << ":" 
				 << r1 << ":" << rrange.first << ":" 
				 << rrange.second  << " Z " << z2 << ":" << z1
				 << ":" << zrange.first << ":" << zrange.second
				 << " Match " << match << ":" << inok << ":" 
				 << outok << " " << ck;
      edm::LogVerbatim("HGCSim") << "Original " << pos.x() << ":" << pos.y()
				 << " return " << xy.first << ":" << xy.second;
      if (DetId(index).det() == DetId::HGCalHSi) {
	double wt=0, xx = ((pos.z() > 0) ? pos.x() : -pos.x());
	int    waferU,waferV,cellU,cellV,waferType;
	hgcons_.waferFromPosition(xx,pos.y(),lay,waferU,waferV,cellU,
				  cellV,waferType,wt,true);
	xy = hgcons_.locateCell(lay,waferU,waferV,cellU,cellV,false,true,true);
	edm::LogVerbatim("HGCSim") << "HGCalNumberingScheme " 
				   << HGCSiliconDetId(index) << " position "
				   << xy.first << ":" << xy.second;
      }
    }
  }
}

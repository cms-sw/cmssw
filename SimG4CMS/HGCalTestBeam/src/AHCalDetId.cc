#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const AHCalDetId AHCalDetId::Undefined(0,0,0);

AHCalDetId::AHCalDetId() : DetId() {
}

AHCalDetId::AHCalDetId(uint32_t rawid) : DetId(rawid) {
}

AHCalDetId::AHCalDetId(int row, int col, int depth) {
  int icol  = (col > 0) ? col : 10-col;
  int irow  = (row > 0) ? row : 10-row;
  id_       = HcalDetId(HcalOther,irow,icol,depth).rawId();
}

AHCalDetId::AHCalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || subdet!=HcalOther) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize AHCalDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_=gen.rawId();
}

int AHCalDetId::irow() const {
  int value = HcalDetId(id_).ietaAbs();
  if (value >= 10) value = -(value%10);
  return value;
}
  
int AHCalDetId::icol() const { 
  int value = HcalDetId(id_).iphi();
  if (value >= 10) value = -(value%10);
  return value;
}

int AHCalDetId::depth() const {
  return HcalDetId(id_).depth();
}

std::pair<double,double> AHCalDetId::getXY() const {
  int row = irow();
  int col = icol();
  double shiftx = (col > 0) ? -0.5*deltaX_ : 0.5*deltaX_;
  double shifty = (row > 0) ? -0.5*deltaY_ : 0.5*deltaY_;
  return std::pair<double,double>(col*deltaX_+shiftx,row*deltaY_+shifty);
}

double AHCalDetId::getZ() const {
  int lay = depth();
  return (zFirst_+(lay-1)*deltaZ_);
}

std::ostream& operator<<(std::ostream& s,const AHCalDetId& id) {
  return s << "(AHCal " << id.irow() << ',' << id.icol() << ',' << id.depth() << ')';
}



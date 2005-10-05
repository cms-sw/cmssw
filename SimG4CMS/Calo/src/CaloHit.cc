///////////////////////////////////////////////////////////////////////////////
// File: CaloHit.cc
// Description: Temporary Hit class for QIE Analysis
///////////////////////////////////////////////////////////////////////////////
//#include "Utilities/Configuration/interface/Architecture.h"
#include "SimG4CMS/Calo/interface/CaloHit.h"

#include <iomanip>

//UserVerbosity CaloHit::cout("CaloHit","silent","CaloSD");

CaloHit::CaloHit(int deti, int layi, double ei, double etai, double fi, 
		 double timi, unsigned int idi): deth(deti), layerh(layi), 
						 eh(ei), etah(etai), phih(fi), 
						 timeh(timi), idh(idi) {}

CaloHit::CaloHit():  deth(0), layerh(0), eh(0), etah(0), phih(0), timeh(0),
		     idh(0) {}

CaloHit::CaloHit(const CaloHit &right) {
  deth   = right.deth;
  layerh = right.layerh;
  eh     = right.eh;
  etah   = right.etah;
  phih   = right.phih;
  timeh  = right.timeh;
  idh    = right.idh;
}

CaloHit::~CaloHit() {}

bool CaloHit::operator<(const CaloHit& hit) const {
  return (eh/cosh(etah) < hit.e()/cosh(hit.eta())) ? false : true ; 
}

void CaloHit::print() {
  std::cout << "CaloHit:: " << (*this);
}

std::ostream& operator<<(std::ostream& os, const CaloHit& hit) {
  os << "E "     << std::setw(6) << hit.e()   << " eta " << std::setw(6) << hit.eta() 
     << " phi "  << std::setw(6) << hit.phi() << " t "   << std::setw(6) << hit.t() 
     << " layer " << hit.layer() << " det " << hit.det() << " id 0x" << std::hex 
     << hit.id() << std::dec;
  return os;
}

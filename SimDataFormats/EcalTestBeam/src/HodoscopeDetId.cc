#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
HodoscopeDetId::HodoscopeDetId() : DetId() {
}
  
HodoscopeDetId::HodoscopeDetId(uint32_t rawid) : DetId(rawid) {
}
  
// use the LaserPnDiode as sub-detector to avoid to create a new one

HodoscopeDetId::HodoscopeDetId(int indexPlane, int indexFibr) 
  : DetId(Ecal,EcalLaserPnDiode)
{
  int iPlane = indexPlane;
  int iFibr = indexFibr;
  if (iPlane < MIN_PLANE || iPlane > MAX_PLANE ||
      iFibr < MIN_FIBR || iFibr > MAX_FIBR) {
    throw cms::Exception("InvalidDetId") << "HodoscopeDetId:  Cannot create object.  Indexes out of bounds.";
  }
  id_ |= ((iPlane&0x3) | ((iFibr&0x3F)<<2)) ;
}
  
HodoscopeDetId::HodoscopeDetId(const DetId& gen) {
 if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalLaserPnDiode )) {
    throw cms::Exception("InvalidDetId");
  }
  id_=gen.rawId();
}
  
HodoscopeDetId& HodoscopeDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalLaserPnDiode )) {
    throw cms::Exception("InvalidDetId");
  }
  id_=gen.rawId();
  return *this;
}
  
std::ostream& operator<<(std::ostream& s,const HodoscopeDetId& id) {
  return s << "(Plane " << id.planeId() << ", fiber " << id.fibrId() << ')';
}
  

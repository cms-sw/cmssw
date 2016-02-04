#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"

std::ostream& operator<<(std::ostream& s, const EcalTBHodoscopeRecInfo& recoHodo) {
  s << "Reconstructed position x: " << recoHodo.posX() << " y: " << recoHodo.posY() 
    << " xSplope: " << recoHodo.slopeX() << " ySplope: " << recoHodo.slopeY() 
    << " xQuality: " << recoHodo.qualX() << " yQuality: " << recoHodo.qualY(); 
  return s;
}

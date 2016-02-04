#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"

std::ostream& operator<<(std::ostream& s, const EcalTBHodoscopeRawInfo& planeHits) {
  s << "Number of planes: " << planeHits.planes() << std::endl;
  for (unsigned int i=0; i<planeHits.planes(); i++) 
    s << planeHits[i] << std::endl;
  return s;
}

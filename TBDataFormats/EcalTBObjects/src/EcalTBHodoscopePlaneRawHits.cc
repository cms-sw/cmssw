#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"

std::ostream& operator<<(std::ostream& s, const EcalTBHodoscopePlaneRawHits& planeHits) {
  s << "Number of channels: " << planeHits.channels() << std::endl;
  for (unsigned int i=0; i<planeHits.channels(); i++) 
    s << "Channel " << i << ": " << (int)planeHits[i] << std::endl;
  return s;
}

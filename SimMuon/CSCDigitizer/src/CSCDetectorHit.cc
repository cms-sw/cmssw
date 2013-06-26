#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include <iostream>

std::ostream & operator<<(std::ostream & stream, const CSCDetectorHit & hit) {
  stream << "element: " << hit.getElement()
         << "  charge: " << hit.getCharge()
         << "   pos:  " << hit.getPosition()
         << "   time: " << hit.getTime() << std::endl; 
  return stream;
}


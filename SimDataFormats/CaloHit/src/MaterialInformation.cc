#include "SimDataFormats/CaloHit/interface/MaterialInformation.h"
#include <iomanip>
#include <iostream>

std::ostream& operator<<(std::ostream& o, const MaterialInformation& info) {
  o << info.vname() << " ID " << info.id() << " Eta:Phi " << info.trackEta() << ":" << info.trackPhi()
    << " Step Length " << std::setw(10) << info.stepLength() << " Radiation Length " << std::setw(10)
    << info.radiationLength() << " Interaction Length " << std::setw(10) << info.interactionLength();

  return o;
}

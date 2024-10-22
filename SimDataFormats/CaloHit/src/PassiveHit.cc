#include "SimDataFormats/CaloHit/interface/PassiveHit.h"
#include <iostream>

std::ostream& operator<<(std::ostream& o, const PassiveHit& hit) {
  o << hit.vname() << "  0x" << std::hex << hit.id() << std::dec << ": Energy " << hit.energy()
    << " GeV: " << hit.energyTotal() << " GeV: Tof " << hit.time() << " ns: "
    << " Track # " << hit.trackId() << " Position " << hit.x() << ", " << hit.y() << ", " << hit.z() << ")";

  return o;
}

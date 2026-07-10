#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <ostream>

namespace io_v1 {
  std::ostream& operator<<(std::ostream& o, const PSimHit& hit) {
    return o << hit.detUnitId() << " " << hit.entryPoint() << " " << hit.tof();
  }
}  // namespace io_v1

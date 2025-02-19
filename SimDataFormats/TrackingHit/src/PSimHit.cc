#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <ostream>

std::ostream & operator<<(std::ostream & o, const PSimHit & hit) 
{ return o << hit.detUnitId() << " " << hit.entryPoint() << " " << hit.tof(); }


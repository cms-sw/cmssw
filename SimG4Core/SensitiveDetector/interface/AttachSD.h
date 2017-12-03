#ifndef SimG4Core_SensitiveDetector_AttachSD_h
#define SimG4Core_SensitiveDetector_AttachSD_h


#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include <vector>

namespace edm {
  class ParameterSet;
}
class SensitiveTkDetector;
class SensitiveCaloDetector;
class SimActivityRegistry;
class SimTrackManager;

class AttachSD
{
public:
  AttachSD();
  ~AttachSD();

  std::pair< std::vector<SensitiveTkDetector*>,
    std::vector<SensitiveCaloDetector*> > 
    create(const DDDWorld &, const DDCompactView &,
	   const SensitiveDetectorCatalog &,
	   edm::ParameterSet const &,
	   const SimTrackManager*,
	   SimActivityRegistry& reg ) const;
};

#endif

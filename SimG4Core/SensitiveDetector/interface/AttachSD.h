#ifndef SimG4Core_AttachSD_h
#define SimG4Core_AttachSD_h


#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include <vector>

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
      std::vector<SensitiveCaloDetector*> > create(const DDDWorld & w, 
						   const DDCompactView & cpv,
						   SensitiveDetectorCatalog & clg,
						   edm::ParameterSet const & p,
						   const SimTrackManager* m,
						   SimActivityRegistry& reg ) const;
};

#endif

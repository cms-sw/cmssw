#ifndef SimG4Core_AttachSD_h
#define SimG4Core_AttachSD_h


#include "SimG4Core/Geometry/interface/DDDWorld.h"
#include <vector>

class SensitiveDetector;

class AttachSD
{
public:
    AttachSD();
    ~AttachSD();
    std::vector<SensitiveDetector*> create(const DDDWorld & w, 
					   const DDCompactView & cpv,
					   edm::ParameterSet const & p) const;
};

#endif

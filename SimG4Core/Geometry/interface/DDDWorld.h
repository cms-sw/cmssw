#ifndef SimG4Core_DDDWorld_h
#define SimG4Core_DDDWorld_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DDG4Builder;
class G4VPhysicalVolume;
    
class DDDWorld
{
public:
    DDDWorld(const edm::ParameterSet & p);
    ~DDDWorld();
    void SetAsWorld(G4VPhysicalVolume * pv);
private:
    DDG4Builder * theBuilder;
};

#endif

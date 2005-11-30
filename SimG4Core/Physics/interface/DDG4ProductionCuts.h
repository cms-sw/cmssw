#ifndef SimG4Core_DDG4ProductionCuts_H
#define SimG4Core_DDG4ProductionCuts_H

#include "SimG4Core/Notification/interface/Observer.h"

#include "SimG4Core/Geometry/interface/DDDWorld.h"

#include <string>
#include <vector>

class DDLogicalPart;
class G4Region;
class G4LogicalVolume;
class G4ProductionCuts;

/**
 * Observes DDDWorld dispatching and reads/sets production cuts.
 * This is the NEW way of doing it, via cuts per region.
 */
class DDG4ProductionCuts : public Observer <const DDDWorld *> 
{
public:
    DDG4ProductionCuts();
    ~DDG4ProductionCuts();
private:
    void update(const DDDWorld * world);
    void SetProdCuts(const DDLogicalPart lpart, G4LogicalVolume * lvolume);
    G4Region * GetRegion(const std::string & region);
    G4ProductionCuts * GetProductionCuts(G4Region * region);
    std::string keywordRegion;
};

#endif

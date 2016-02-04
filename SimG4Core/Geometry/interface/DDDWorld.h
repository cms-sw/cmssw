#ifndef SimG4Core_DDDWorld_h
#define SimG4Core_DDDWorld_h

#include "SimG4Core/Geometry/interface/DDGeometryReturnType.h"

class DDG4Builder;
class G4VPhysicalVolume;
class DDCompactView;    

class DDDWorld {

public:
  DDDWorld(const DDCompactView*, G4LogicalVolumeToDDLogicalPartMap &,
	   SensitiveDetectorCatalog &, bool check=false);
  ~DDDWorld();
  void SetAsWorld(G4VPhysicalVolume * pv);

private:
  DDG4Builder * theBuilder;
};

#endif

#ifndef SimG4Core_DDDWorld_h
#define SimG4Core_DDDWorld_h

#include "SimG4Core/Geometry/interface/DDGeometryReturnType.h"
#include "G4VPhysicalVolume.hh"

class DDG4Builder;
class DDCompactView;    

class DDDWorld {

public:
  DDDWorld(const DDCompactView*, G4LogicalVolumeToDDLogicalPartMap &,
	   SensitiveDetectorCatalog &, bool check=false);
  ~DDDWorld();
  void SetAsWorld(G4VPhysicalVolume * pv);
  const G4VPhysicalVolume * GetWorldVolume() const { return m_world; }

private:
  DDG4Builder * theBuilder;
  G4VPhysicalVolume * m_world;
};

#endif

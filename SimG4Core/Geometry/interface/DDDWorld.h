#ifndef SimG4Core_DDDWorld_h
#define SimG4Core_DDDWorld_h

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "G4VPhysicalVolume.hh"

class DDCompactView;

namespace cms {
  class DDCompactView;
}

class DDDWorld {
public:
  DDDWorld(const DDCompactView *pDD,
           const cms::DDCompactView *pDD4hep,
           SensitiveDetectorCatalog &,
           int verb,
           bool cuts,
           bool pcut);
  DDDWorld(const DDCompactView *, G4LogicalVolumeToDDLogicalPartMap &, SensitiveDetectorCatalog &, bool check);
  ~DDDWorld();
  G4VPhysicalVolume *GetWorldVolume() const { return m_world; }

private:
  G4VPhysicalVolume *m_world;
};

#endif

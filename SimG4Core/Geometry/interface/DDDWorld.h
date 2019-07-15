#ifndef SimG4Core_DDDWorld_h
#define SimG4Core_DDDWorld_h

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "G4VPhysicalVolume.hh"
#include "DDG4/Geant4GeometryInfo.h"

class DDG4Builder;
class DDCompactView;

namespace cms {
  class DDDetector;
}  // namespace cms

class DDDWorld {
public:
  DDDWorld(const DDCompactView *, G4LogicalVolumeToDDLogicalPartMap &, SensitiveDetectorCatalog &, bool check);
  DDDWorld(const cms::DDDetector *, dd4hep::sim::Geant4GeometryMaps::VolumeMap &);
  ~DDDWorld();
  G4VPhysicalVolume *GetWorldVolume() const { return m_world; }

private:
  G4VPhysicalVolume *m_world;
};

#endif

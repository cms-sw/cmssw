#ifndef SimG4Core_DDDWorld_h
#define SimG4Core_DDDWorld_h

#include "G4VPhysicalVolume.hh"
#include "SimG4Core/Geometry/interface/DDGeometryReturnType.h"
#include "DDG4/Geant4GeometryInfo.h"

class DDG4Builder;
class DDCompactView;

namespace cms {
  class DDDetector;
}  // namespace cms

class DDDWorld {
public:
  DDDWorld(const DDCompactView *, G4LogicalVolumeToDDLogicalPartMap &, SensitiveDetectorCatalog &, bool check = false);
  DDDWorld(const cms::DDDetector *, dd4hep::sim::Geant4GeometryMaps::VolumeMap &);
  ~DDDWorld();
  G4VPhysicalVolume *GetWorldVolume() const { return m_world; }

private:
  G4VPhysicalVolume *m_world;
};

#endif

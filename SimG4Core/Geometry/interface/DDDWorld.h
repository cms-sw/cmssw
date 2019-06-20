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
  static void WorkerSetAsWorld(G4VPhysicalVolume *pv);
  const G4VPhysicalVolume *GetWorldVolume() const { return m_world; }

  // In order to share the world volume with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4
  // with TLS. Should we consider a friend declaration here in order
  // to avoid misuse?
  G4VPhysicalVolume *GetWorldVolumeForWorker() const { return m_world; }

private:
  void SetAsWorld(G4VPhysicalVolume *pv);
  G4VPhysicalVolume *m_world;
};

#endif

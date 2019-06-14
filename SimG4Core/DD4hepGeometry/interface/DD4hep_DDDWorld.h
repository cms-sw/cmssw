#ifndef SIMG4CORE_DD4HEP_DDDWORLD_H
#define SIMG4CORE_DD4HEP_DDDWORLD_H

#include "G4VPhysicalVolume.hh"
#include "DDG4/Geant4GeometryInfo.h"

namespace cms {
  class DDDetector;
}

namespace cms {
  class DDDWorld {
  public:
    DDDWorld(const cms::DDDetector*, dd4hep::sim::Geant4GeometryMaps::VolumeMap&);
    ~DDDWorld();
    static void workerSetAsWorld(G4VPhysicalVolume* pv);
    const G4VPhysicalVolume* getWorldVolume() const { return m_world; }

    // In order to share the world volume with the worker threads, we
    // need a non-const pointer. Thread-safety is handled inside Geant4
    // with TLS. Should we consider a friend declaration here in order
    // to avoid misuse?
    G4VPhysicalVolume* getWorldVolumeForWorker() const { return m_world; }

  private:
    void setAsWorld(G4VPhysicalVolume* pv);
    G4VPhysicalVolume* m_world;
  };
}  // namespace cms

#endif

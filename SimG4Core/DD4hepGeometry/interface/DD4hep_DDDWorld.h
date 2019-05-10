#ifndef SIMG4CORE_DD4HEP_DDDWORLD_H
#define SIMG4CORE_DD4HEP_DDDWORLD_H

#include "G4VPhysicalVolume.hh"

namespace cms {

  class DDDetector;
  
  class DDDWorld {

  public:
    DDDWorld(const cms::DDDetector*);
    ~DDDWorld();
    static void SetAsWorld(G4VPhysicalVolume* pv);
    static void WorkerSetAsWorld(G4VPhysicalVolume* pv);
    const G4VPhysicalVolume* GetWorldVolume() const { return m_world; }
    
    // In order to share the world volume with the worker threads, we
    // need a non-const pointer. Thread-safety is handled inside Geant4
    // with TLS. Should we consider a friend declaration here in order
    // to avoid misuse?
    G4VPhysicalVolume* GetWorldVolumeForWorker() const { return m_world; }

  private:
    G4VPhysicalVolume* m_world;
  };
}

#endif

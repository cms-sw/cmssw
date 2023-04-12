#ifndef SimG4Core_PhysicsLists_CMSHepEmTrackingManager_h
#define SimG4Core_PhysicsLists_CMSHepEmTrackingManager_h

#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1100

#include "G4HepEmTrackingManager.hh"

class CMSHepEmTrackingManager final : public G4HepEmTrackingManager {
public:
  CMSHepEmTrackingManager(G4double highEnergyLimit);
  ~CMSHepEmTrackingManager();

  void BuildPhysicsTable(const G4ParticleDefinition &) override;

  void PreparePhysicsTable(const G4ParticleDefinition &) override;

  void HandOverOneTrack(G4Track *aTrack) override;

private:
  G4double fHighEnergyLimit;
};

#endif

#endif

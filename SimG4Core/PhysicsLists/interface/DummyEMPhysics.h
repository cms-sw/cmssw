#ifndef SimG4Core_PhysicsLists_DummyEMPhysics_h
#define SimG4Core_PhysicsLists_DummyEMPhysics_h

// Physics List equivalent to GeantV

#include "G4VPhysicsConstructor.hh"

class DummyEMPhysics : public G4VPhysicsConstructor {
public:
  DummyEMPhysics(G4int verb);
  ~DummyEMPhysics() override = default;
  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int verbose;
};

#endif

#ifndef SimG4Core_CustomPhysics_Quirk_H
#define SimG4Core_CustomPhysics_Quirk_H

// Quirk particle definition (lepton-like, infracolor charged).
// Carries no string state: the definition is shared between threads, the
// state lives in the thread-local QuirkStringStore.

#include "G4ParticleDefinition.hh"

class Quirk : public G4ParticleDefinition {
public:
  Quirk(const G4String& name, G4double mass, G4double charge, G4int pdg);
  ~Quirk() override = default;

  Quirk(const Quirk&) = delete;
  Quirk& operator=(const Quirk&) = delete;

  static bool isQuirk(const G4ParticleDefinition* p) { return nullptr != dynamic_cast<const Quirk*>(p); }
};

#endif

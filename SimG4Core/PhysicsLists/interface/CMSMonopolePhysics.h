#ifndef SimG4Core_PhysicsLists_CMSMonopolePhysics_h
#define SimG4Core_PhysicsLists_CMSMonopolePhysics_h

#include "HepPDT/ParticleData.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSMonopolePhysics : public G4VPhysicsConstructor {

public:
  CMSMonopolePhysics(const HepPDT::ParticleData *particle, G4double charge, G4int ver);
  virtual ~CMSMonopolePhysics();

  void ConstructParticle();
  void ConstructProcess();

private:
  G4int    verbose;
  G4bool   ok;
  G4double monopoleMass;
  G4int    elCharge, magCharge, pdgEncoding;
};

#endif







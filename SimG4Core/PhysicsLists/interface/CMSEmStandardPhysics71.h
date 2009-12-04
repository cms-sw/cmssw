#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics71_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics71_h

#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysics71 : public G4VPhysicsConstructor {

public:
  CMSEmStandardPhysics71(const G4String& name, const HepPDT::ParticleDataTable * table_, G4int ver, G4double charge_);
  virtual ~CMSEmStandardPhysics71();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
  CMSMonopolePhysics* monopolePhysics;
};

#endif







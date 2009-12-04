#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics92_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics92_h

#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include <string>

class CMSEmStandardPhysics92 : public G4VPhysicsConstructor {

public:
  CMSEmStandardPhysics92(const G4String& name, const HepPDT::ParticleDataTable * table_, G4int ver, std::string reg, G4double charge_);
  virtual ~CMSEmStandardPhysics92();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
  std::string         region;
  CMSMonopolePhysics* monopolePhysics;
};

#endif







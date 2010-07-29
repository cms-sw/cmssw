#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsSync_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsSync_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include <string>

class CMSEmStandardPhysicsSync : public G4VPhysicsConstructor {

public:
  CMSEmStandardPhysicsSync(const G4String& name, G4int ver, G4bool type, std::string reg);
  virtual ~CMSEmStandardPhysicsSync();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
  G4bool              srType;
  std::string         region;
};

#endif







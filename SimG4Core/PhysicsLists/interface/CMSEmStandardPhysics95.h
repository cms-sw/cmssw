#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics95_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics95_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include <string>

class CMSEmStandardPhysics95 : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysics95(const G4String& name, G4int ver, const std::string& reg);
  ~CMSEmStandardPhysics95() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int               verbose;
  std::string         region;
};

#endif







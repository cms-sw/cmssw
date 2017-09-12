#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics95msc93_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics95msc93_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include <string>

class CMSEmStandardPhysics95msc93 : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysics95msc93(const G4String& name, G4int ver, const std::string& reg);
  ~CMSEmStandardPhysics95msc93() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int               verbose;
  std::string         region;
};

#endif







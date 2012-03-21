#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics95msc92_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics95msc92_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include <string>

class CMSEmStandardPhysics95msc92 : public G4VPhysicsConstructor {

public:
  CMSEmStandardPhysics95msc92(const G4String& name, G4int ver, std::string reg);
  virtual ~CMSEmStandardPhysics95msc92();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
  std::string         region;
};

#endif







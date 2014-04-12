#ifndef SimG4Core_PhysicsLists_CMSEmNoDeltaRay_h
#define SimG4Core_PhysicsLists_CMSEmNoDeltaRay_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include <string>

class CMSEmNoDeltaRay : public G4VPhysicsConstructor {

public:
  CMSEmNoDeltaRay(const G4String& name, G4int ver, std::string reg);
  virtual ~CMSEmNoDeltaRay();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
  std::string         region;
};

#endif







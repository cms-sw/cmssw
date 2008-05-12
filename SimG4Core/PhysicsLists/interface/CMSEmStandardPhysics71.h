#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics71_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics71_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysics71 : public G4VPhysicsConstructor {

public:
  CMSEmStandardPhysics71(const G4String& name, G4int ver);
  virtual ~CMSEmStandardPhysics71();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int  verbose;
};

#endif







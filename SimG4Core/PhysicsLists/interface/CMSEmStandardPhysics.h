#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysics : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysics(G4int ver);
  virtual ~CMSEmStandardPhysics();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
};

#endif







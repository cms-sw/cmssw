#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsXS_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsXS_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsXS : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysicsXS(G4int ver);
  virtual ~CMSEmStandardPhysicsXS();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
};

#endif







#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics_h

#include "G4EmStandardPhysics.hh"

class CMSEmStandardPhysics : public G4EmStandardPhysics {

public:
  CMSEmStandardPhysics(const G4String& name, G4int ver);
  virtual ~CMSEmStandardPhysics() {}
};

#endif


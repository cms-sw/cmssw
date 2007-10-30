#include "SimG4Core/PhysicsLists/interface/G4Version.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"

CMSEmStandardPhysics::CMSEmStandardPhysics(const G4String& name, G4int ver) :
#ifndef G4V9
  G4EmStandardPhysics(name, ver) {}
#else
  G4EmStandardPhysics(ver, name) {}
#endif

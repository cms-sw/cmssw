#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics71.h"

CMSEmStandardPhysics71::CMSEmStandardPhysics71(const G4String& name, G4int ver) :
#ifndef G4V9
  G4EmStandardPhysics71(name, ver) {}
#else
  G4EmStandardPhysics_option1(ver, name) {}
#endif

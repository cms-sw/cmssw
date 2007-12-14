#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics71_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics71_h

#include "SimG4Core/PhysicsLists/interface/G4Version.h"
#ifndef G4V9
#include "G4EmStandardPhysics71.hh"

class CMSEmStandardPhysics71 : public G4EmStandardPhysics71 {
#else
#include "G4EmStandardPhysics_option1.hh"

class CMSEmStandardPhysics71 : public G4EmStandardPhysics_option1 {
#endif

public:
  CMSEmStandardPhysics71(const G4String& name, G4int ver);
  virtual ~CMSEmStandardPhysics71() {}
};

#endif


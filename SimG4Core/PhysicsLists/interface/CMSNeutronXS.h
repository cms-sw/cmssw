#ifndef SimG4Core_PhysicsLists_CMSNeutronXS_h
#define SimG4Core_PhysicsLists_CMSNeutronXS_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSNeutronXS : public G4VPhysicsConstructor {

public:

  CMSNeutronXS(const G4String& nam, G4int ver);
  virtual ~CMSNeutronXS();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int  verbose;
};

#endif







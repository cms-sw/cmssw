#ifndef SimG4Core_PhysicsLists_CMSGlauberGribovXS_h
#define SimG4Core_PhysicsLists_CMSGlauberGribovXS_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSGlauberGribovXS : public G4VPhysicsConstructor {

public:

  CMSGlauberGribovXS(G4int ver);
  virtual ~CMSGlauberGribovXS();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int  verbose;
};

#endif







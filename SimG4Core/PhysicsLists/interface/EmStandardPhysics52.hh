#ifndef SimG4Core_PhysicsLists_EmStandardPhysics52_h
#define SimG4Core_PhysicsLists_EmStandardPhysics52_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class EmStandardPhysics52 : public G4VPhysicsConstructor {

public:
  EmStandardPhysics52(const G4String& name = "EMstandard52", G4int ver = 1);
  virtual ~EmStandardPhysics52();

public:
  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int  verbose;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif







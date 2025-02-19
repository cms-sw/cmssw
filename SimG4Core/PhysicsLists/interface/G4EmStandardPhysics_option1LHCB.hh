#ifndef G4EmStandardPhysics_option1LHCB_h
#define G4EmStandardPhysics_option1LHCB_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4EmStandardPhysics_option1LHCB : public G4VPhysicsConstructor
{
public:

  G4EmStandardPhysics_option1LHCB(G4int ver = 1);

  virtual ~G4EmStandardPhysics_option1LHCB();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int  verbose;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif







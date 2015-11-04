//
//  Initial version from Geant4 example
//  exoticphysics/monopole/include/G4MonopoleEquation.hh
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//
//
// class G4MonopoleEquation
//
// Class description:
//
// This is the right-hand side of equation of motion in a combined
// electric and magnetic field for magnetic monopoles.

// History:
// - Created. V.Grichine, 10.11.98
// - Modified. S.Burdin, 30.04.10
//                B.Bozsogi, 15.06.10
// -------------------------------------------------------------------

#ifndef G4MONOPOLEEQUATION_hh
#define G4MONOPOLEEQUATION_hh

#include "G4EquationOfMotion.hh"
#include "G4ElectroMagneticField.hh"

class G4MonopoleEquation : public G4EquationOfMotion
{
public:  // with description

  G4MonopoleEquation(G4ElectroMagneticField *emField );

  ~G4MonopoleEquation();

  virtual void  SetChargeMomentumMass( G4ChargeState particleChargeState,
                                       G4double      momentum, 
                                       G4double      mass);
  // magnetic charge in e+ units
                                 
  virtual void EvaluateRhsGivenB(const G4double y[],
                         const G4double Field[],
                         G4double dydx[] ) const;
  // Given the value of the electromagnetic field, this function 
  // calculates the value of the derivative dydx.

private:

  G4double  fMagCharge ;
  G4double  fElCharge;
  G4double  fMassCof;
};

#endif

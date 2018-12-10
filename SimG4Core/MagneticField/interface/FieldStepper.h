#ifndef SimG4Core_MagneticField_FieldStepper_H
#define SimG4Core_MagneticField_FieldStepper_H

#include "G4MagIntegratorStepper.hh"

class G4Mag_UsualEqRhs;

class FieldStepper : public G4MagIntegratorStepper
{
public:
  explicit FieldStepper(G4Mag_UsualEqRhs * eq, double del, const std::string& name);
  ~FieldStepper() override;

  // Geant4 virtual methods
  void Stepper(const G4double y[], const G4double dydx[], G4double h,
	       G4double yout[], G4double yerr[]) override;
  G4double DistChord() const override;
  G4int IntegratorOrder() const override;

private:
  void selectStepper(const std::string&);

  G4MagIntegratorStepper * theStepper;
  G4Mag_UsualEqRhs * theEquation;  
  double theDelta;
};

#endif

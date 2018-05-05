//
// =======================================================================
//
// class G4MonopoleEquation 
//
// Created:  30 April 2010, S. Burdin, B. Bozsogi 
//                       G4MonopoleEquation class for
//                       Geant4 extended example "monopole"
//
// Adopted for CMSSW by V.Ivanchenko 30 April 2018
// from Geant4 global tag geant4-10-04-ref-03                   
//
// =======================================================================
//
// Class description:
//
// This is the right-hand side of equation of motion in a 
// magnetic field only for magnetic monopoles.
//
// -------------------------------------------------------------------

#ifndef SimG4Core_MagneticField_MonopoleEquation_h
#define SimG4Core_MagneticField_MonopoleEquation_h 1

#include "G4EquationOfMotion.hh"
#include "G4MagneticField.hh"

class MonopoleEquation : public G4EquationOfMotion
{
public:  // with description

  MonopoleEquation(G4MagneticField *emField );

  ~MonopoleEquation() override;

  void  SetChargeMomentumMass( G4ChargeState particleChargeState,
                               G4double      momentum, 
                               G4double      mass) override;
  // magnetic charge in e+ units
                                 
  void EvaluateRhsGivenB(const G4double y[],
                         const G4double Field[],
                         G4double dydx[] ) const override;
  // Given the value of the electromagnetic field, this function 
  // calculates the value of the derivative dydx.

private:

  G4double  fMagCharge ;
  G4double  fElCharge;
  G4double  fMassCof;
};

#endif

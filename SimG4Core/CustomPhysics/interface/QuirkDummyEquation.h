#ifndef SimG4Core_CustomPhysics_QuirkDummyEquation_H
#define SimG4Core_CustomPhysics_QuirkDummyEquation_H

// Empty equation of motion, only needed to construct QuirkHyperbolaStepper.

#include "G4EquationOfMotion.hh"
#include "G4UniformMagField.hh"

#include <memory>

class QuirkDummyEquation : public G4EquationOfMotion {
public:
  QuirkDummyEquation() : G4EquationOfMotion(nullptr), m_dummyField(std::make_unique<G4UniformMagField>(0, 0, 0)) {
    SetFieldObj(m_dummyField.get());
  }
  ~QuirkDummyEquation() override = default;

  void EvaluateRhsGivenB(const G4double[], const G4double[3], G4double[]) const override {}
  void SetChargeMomentumMass(G4ChargeState, G4double, G4double) override {}

private:
  std::unique_ptr<G4UniformMagField> m_dummyField;
};

#endif

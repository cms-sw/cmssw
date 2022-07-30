//
// copied from G4TMagFieldEquation
//
// Class description:
//
// Templated version of equation of motion of a particle in a pure magnetic field.
// Enables use of inlined code for field, equation, stepper, driver,
// avoiding all virtual calls.
//
// Adapted from G4Mag_UsualEqRhs.hh
// --------------------------------------------------------------------
// Created: Josh Xie  (Google Summer of Code 2014 )
// Adapted from G4Mag_UsualEqRhs
//
// Modified by Vincenzo innocente 7/2022 to adapt to specific CMS field
//
#ifndef CMSTMAGFIELDEQUATION_H
#define CMSTMAGFIELDEQUATION_H

#include "G4Mag_UsualEqRhs.hh"
#include <cassert>

template <class T_Field>
class CMSTMagFieldEquation final : public G4Mag_UsualEqRhs {
public:
  CMSTMagFieldEquation(T_Field* f) : G4Mag_UsualEqRhs(f) {
    assert(f);
    itsField = f;
  }

  ~CMSTMagFieldEquation() override { ; }

  inline void GetFieldValueCMS(const G4double Point[], G4double Field[]) const {
    itsField->GetFieldValue(Point, Field);
  }

  inline void TEvaluateRhsGivenB(const G4double y[],
                                 G4double inv_momentum_magnitude,
                                 const G4double B[3],
                                 G4double dydx[]) const {
    G4double cof = FCof() * inv_momentum_magnitude;

    dydx[0] = y[3] * inv_momentum_magnitude;  //  (d/ds)x = Vx/V
    dydx[1] = y[4] * inv_momentum_magnitude;  //  (d/ds)y = Vy/V
    dydx[2] = y[5] * inv_momentum_magnitude;  //  (d/ds)z = Vz/V

    dydx[3] = cof * (y[4] * B[2] - y[5] * B[1]);  // Ax = a*(Vy*Bz - Vz*By)
    dydx[4] = cof * (y[5] * B[0] - y[3] * B[2]);  // Ay = a*(Vz*Bx - Vx*Bz)
    dydx[5] = cof * (y[3] * B[1] - y[4] * B[0]);  // Az = a*(Vx*By - Vy*Bx)

    return;
  }

  __attribute__((always_inline)) void TRightHandSide(const G4double y[],
                                                     G4double inv_momentum_magnitude,
                                                     G4double dydx[]) const {
    // CMS field  is three dimentional
    G4double Field[3];
    GetFieldValueCMS(y, Field);
    TEvaluateRhsGivenB(y, inv_momentum_magnitude, Field, dydx);
  }

private:
  // Dependent objects
  T_Field* itsField;
};

#endif

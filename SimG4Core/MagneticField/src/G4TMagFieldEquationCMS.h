//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// G4TMagFieldEquation
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

#include "G4Mag_UsualEqRhs.hh"
#include <cassert>

template <class T_Field>
class G4TMagFieldEquationCMS final : public G4Mag_UsualEqRhs {
public:
  G4TMagFieldEquationCMS(T_Field* f) : G4Mag_UsualEqRhs(f) {
    assert(f);
    itsField = f;
  }

  virtual ~G4TMagFieldEquationCMS() { ; }

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

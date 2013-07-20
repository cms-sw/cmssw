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
// $Id: G4MonopoleEquation.hh,v 1.1 2010/07/29 22:46:28 sunanda Exp $
// GEANT4 tag $Name: CMSSW_6_2_0 $
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
// 	       B.Bozsogi, 15.06.10
// -------------------------------------------------------------------

#ifndef G4MONOPOLEEQUATION_hh
#define G4MONOPOLEEQUATION_hh

#include "G4EquationOfMotion.hh"
#include "G4ElectroMagneticField.hh"

class G4MonopoleEquation : public G4EquationOfMotion
{
  public:  // with description

  G4MonopoleEquation(G4ElectroMagneticField *emField );
      // : G4EquationOfMotion( emField ) {;}

    ~G4MonopoleEquation() {;} 

    void  SetChargeMomentumMass(G4double particleMagneticCharge,
                                G4double particleElectricCharge,
                                G4double mass);
                                 
    void EvaluateRhsGivenB(const G4double y[],
                           const G4double Field[],
                                 G4double dydx[] ) const;
      // Given the value of the electromagnetic field, this function 
      // calculates the value of the derivative dydx.

  private:

    G4double        fMagCharge ;
    G4double        fElCharge;
    G4double        fMassCof;
};

#endif

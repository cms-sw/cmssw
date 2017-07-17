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
/// \file exoticphysics/monopole/src/G4MonopoleEquation.cc
/// \brief Implementation of the G4MonopoleEquation class
//
// $Id: G4MonopoleEquation.cc 69705 2013-05-13 09:09:52Z gcosmo $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//
//
// class G4MonopoleEquation
//
// Class description:
//
//
//  This is the standard right-hand side for equation of motion.
//
//  The only case another is required is when using a moving reference
//  frame ... or extending the class to include additional Forces,
//  eg an electric field
//
//  10.11.98   V.Grichine
//
//  30.04.10   S.Burdin (modified to use for the monopole trajectories).
//
//  15.06.10   B.Bozsogi (replaced the hardcoded magnetic charge with
//                        the one passed by G4MonopoleTransportation)
//                       +workaround to pass the electric charge.
// 
//  12.07.10  S.Burdin (added equations for the electric charges)
// -------------------------------------------------------------------

#include "SimG4Core/MagneticField/interface/G4MonopoleEquation.hh"
#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include <iomanip>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4MonopoleEquation::G4MonopoleEquation(G4ElectroMagneticField *emField )
      : G4EquationOfMotion( emField ) 
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4MonopoleEquation::~G4MonopoleEquation()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void  
G4MonopoleEquation::SetChargeMomentumMass( G4ChargeState particleChargeState, 
                                           G4double      ,           // momentum, 
                                           G4double particleMass)
{
   G4double particleMagneticCharge= particleChargeState.MagneticCharge(); 
   G4double particleElectricCharge= particleChargeState.GetCharge(); 

  //   fElCharge = particleElectricCharge;
  fElCharge =eplus* particleElectricCharge*c_light;
   
  fMagCharge =  eplus*particleMagneticCharge*c_light ;

  // G4cout << " G4MonopoleEquation: ElectricCharge=" << particleElectricCharge
  //           << "; MagneticCharge=" << particleMagneticCharge
  //           << G4endl;
 
  fMassCof = particleMass*particleMass ; 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
G4MonopoleEquation::EvaluateRhsGivenB(const G4double y[],
                                      const G4double Field[],
                                      G4double dydx[] ) const
{
  // Components of y:
  //    0-2 dr/ds, 
  //    3-5 dp/ds - momentum derivatives 

  G4double pSquared = y[3]*y[3] + y[4]*y[4] + y[5]*y[5] ;

  G4double Energy   = std::sqrt( pSquared + fMassCof );
   
  G4double pModuleInverse  = 1.0/std::sqrt(pSquared);

  G4double inverse_velocity = Energy * pModuleInverse / c_light;

  G4double cofEl     = fElCharge * pModuleInverse ;
  G4double cofMag = fMagCharge * Energy * pModuleInverse; 


  dydx[0] = y[3]*pModuleInverse ;                         
  dydx[1] = y[4]*pModuleInverse ;                         
  dydx[2] = y[5]*pModuleInverse ;                    
     
  // G4double magCharge = twopi * hbar_Planck / (eplus * mu0);    
  // magnetic charge in SI units A*m convention
  //  see http://en.wikipedia.org/wiki/Magnetic_monopole   
  //   G4cout  << "Magnetic charge:  " << magCharge << G4endl;   
  // dp/ds = dp/dt * dt/ds = dp/dt / v = Force / velocity
  // dydx[3] = fMagCharge * Field[0]  * inverse_velocity  * c_light;    
  // multiplied by c_light to convert to MeV/mm
  //     dydx[4] = fMagCharge * Field[1]  * inverse_velocity  * c_light; 
  //     dydx[5] = fMagCharge * Field[2]  * inverse_velocity  * c_light; 
      
  dydx[3] = cofMag * Field[0] + cofEl * (y[4]*Field[2] - y[5]*Field[1]);   
  dydx[4] = cofMag * Field[1] + cofEl * (y[5]*Field[0] - y[3]*Field[2]); 
  dydx[5] = cofMag * Field[2] + cofEl * (y[3]*Field[1] - y[4]*Field[0]); 
   
  //        G4cout << std::setprecision(5)<< "E=" << Energy
  //               << "; p="<< 1/pModuleInverse
  //               << "; mC="<< magCharge
  //               <<"; x=" << y[0]
  //               <<"; y=" << y[1]
  //               <<"; z=" << y[2]
  //               <<"; dydx[3]=" << dydx[3]
  //               <<"; dydx[4]=" << dydx[4]
  //               <<"; dydx[5]=" << dydx[5]
  //               << G4endl;

  dydx[6] = 0.;//not used
   
  // Lab Time of flight
  dydx[7] = inverse_velocity;
  return;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

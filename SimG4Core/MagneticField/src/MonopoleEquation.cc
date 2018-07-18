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
//
// =======================================================================
//

#include "SimG4Core/MagneticField/interface/MonopoleEquation.h"
#include "globals.hh"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include <iomanip>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

MonopoleEquation::MonopoleEquation(G4MagneticField *emField )
      : G4EquationOfMotion( emField ) 
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

MonopoleEquation::~MonopoleEquation()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void  
MonopoleEquation::SetChargeMomentumMass( G4ChargeState particleChargeState, 
                                         G4double, 
                                         G4double particleMass)
{
  G4double particleMagneticCharge= particleChargeState.MagneticCharge(); 
  G4double particleElectricCharge= particleChargeState.GetCharge(); 

  fElCharge = CLHEP::eplus*particleElectricCharge*CLHEP::c_light;
   
  fMagCharge = CLHEP::eplus*particleMagneticCharge*CLHEP::c_light;

  // G4cout << " MonopoleEquation: ElectricCharge=" << particleElectricCharge
  //           << "; MagneticCharge=" << particleMagneticCharge
  //           << G4endl;
 
  fMassCof = particleMass*particleMass ; 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
MonopoleEquation::EvaluateRhsGivenB(const G4double y[],
                                    const G4double Field[],
                                    G4double dydx[] ) const
{
  // Components of y:
  //    0-2 dr/ds, 
  //    3-5 dp/ds - momentum derivatives 

  G4double pSquared = y[3]*y[3] + y[4]*y[4] + y[5]*y[5] ;

  G4double Energy   = std::sqrt( pSquared + fMassCof );
   
  G4double pModuleInverse  = 1.0/std::sqrt(pSquared);

  G4double inverse_velocity = Energy * pModuleInverse / CLHEP::c_light;

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

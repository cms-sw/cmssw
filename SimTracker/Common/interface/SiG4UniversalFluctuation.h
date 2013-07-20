//
// ********************************************************************
// * DISCLAIMER                                                       *
// *                                                                  *
// * The following disclaimer summarizes all the specific disclaimers *
// * of contributors to this software. The specific disclaimers,which *
// * govern, are listed with their locations in:                      *
// *   http://cern.ch/geant4/license                                  *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.                                                             *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * GEANT4 collaboration.                                            *
// * By copying,  distributing  or modifying the Program (or any work *
// * based  on  the Program)  you indicate  your  acceptance of  this *
// * statement, and all its terms.                                    *
// ********************************************************************
//
// $Id: SiG4UniversalFluctuation.h,v 1.4 2011/06/13 07:18:21 innocent Exp $
// GEANT4 tag $Name: CMSSW_6_2_0 $
//
// -------------------------------------------------------------------
//
// GEANT4 Class header file
//
//
// File name:     G4UniversalFluctuation
//
// Author:        Vladimir Ivanchenko
//
// Creation date: 03.01.2002
//
// Modifications:
//
// 09-12-02 remove warnings (V.Ivanchenko)
// 28-12-02 add method Dispersion (V.Ivanchenko)
// 07-02-03 change signature (V.Ivanchenko)
// 13-02-03 Add name (V.Ivanchenko)
// 16-10-03 Changed interface to Initialisation (V.Ivanchenko)
// 07-02-05 define problim = 5.e-3 (mma)
//
// Modified for standalone use in CMSSW. danek k. 2/06
//
// Class Description:
//
// Implementation of energy loss fluctuations

// -------------------------------------------------------------------
//

#ifndef SiG4UniversalFluctuation_h
#define SiG4UniversalFluctuation_h 

namespace CLHEP{
  class HepRandomEngine;
  class RandPoissonQ;
  class RandGaussQ;
  class RandFlat;
}

//#include "G4VEmFluctuationModel.hh"

class SiG4UniversalFluctuation {
public:

  SiG4UniversalFluctuation(CLHEP::HepRandomEngine&);

  ~SiG4UniversalFluctuation();

  // momentum in MeV/c, mass in MeV, tmax (delta cut) in MeV,
  // length in mm, meanLoss eloss in MeV.
  double SampleFluctuations(const double momentum,
                            const double mass,
                            double& tmax,
                            const double length,
                            const double meanLoss);
   
  //G4double SampleFluctuations(const G4Material*,
  //                      const G4DynamicParticle*,
  //			G4double&,
  //                            G4double&,
  //                            G4double&);

  //G4double Dispersion(    const G4Material*,
  //                      const G4DynamicParticle*,
  //			G4double&,
  //                           G4double&);
  //void InitialiseMe(const G4ParticleDefinition*);

protected:

private:

  CLHEP::HepRandomEngine& rndEngine;
  CLHEP::RandGaussQ* gaussQDistribution;
  CLHEP::RandPoissonQ* poissonQDistribution;
  CLHEP::RandFlat* flatDistribution;
  // hide assignment operator
  //SiG4UniversalFluctuation & operator=(const  SiG4UniversalFluctuation &right);
  //SiG4UniversalFluctuation(const  SiG4UniversalFluctuation&);

  //const G4ParticleDefinition* particle;
  //const G4Material* lastMaterial;

  double particleMass;
  double chargeSquare;

  // data members to speed up the fluctuation calculation
  double ipotFluct;
  double electronDensity;
  //  double zeff;
  
  double f1Fluct;
  double f2Fluct;
  double e1Fluct;
  double e2Fluct;
  double rateFluct;
  double e1LogFluct;
  double e2LogFluct;
  double ipotLogFluct;
  double e0;

  double minNumberInteractionsBohr;
  double theBohrBeta2;
  double minLoss;
  double problim;
  double sumalim;
  double alim;
  double nmaxCont1;
  double nmaxCont2;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif


//
// GEANT4 Class header file
//
//
// File name:   SiG4UniversalFluctuation
//
// Author: Vladimir Ivanchenko make a class for Laszlo Urban model
//
// Modified for standalone use in CMSSW. Danek K. 02/2006
//
// Class Description:
//
// Implementation of energy loss fluctuations in Silicon

// -------------------------------------------------------------------
//

#ifndef SiG4UniversalFluctuation_h
#define SiG4UniversalFluctuation_h

namespace CLHEP {
  class HepRandomEngine;
}

class SiG4UniversalFluctuation {
public:
  explicit SiG4UniversalFluctuation();

  // hide assignment operator
  SiG4UniversalFluctuation &operator=(const SiG4UniversalFluctuation &right) = delete;
  SiG4UniversalFluctuation(const SiG4UniversalFluctuation &) = delete;

  ~SiG4UniversalFluctuation();

  // momentum in MeV/c, mass in MeV, tmax (delta cut) in MeV,
  // length in mm, meanLoss eloss in MeV.
  double SampleFluctuations(const double momentum,
                            const double mass,
                            double &tmax,
                            const double length,
                            const double meanLoss,
                            CLHEP::HepRandomEngine *);

private:
  double particleMass;
  double chargeSquare;

  // data members to speed up the fluctuation calculation
  double ipotFluct;
  double electronDensity;

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

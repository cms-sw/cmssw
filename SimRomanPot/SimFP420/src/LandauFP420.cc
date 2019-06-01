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
// -------------------------------------------------------------------
//
// GEANT4 Class file
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
// 28-12-02 add method Dispersion (V.Ivanchenko)
// 07-02-03 change signature (V.Ivanchenko)
// 13-02-03 Add name (V.Ivanchenko)
// Modified for standalone use in ORCA. d.k. 6/04
//
// -------------------------------------------------------------------

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "SimRomanPot/SimFP420/interface/LandauFP420.h"

#include <cmath>
#include <cstdio>
#include <gsl/gsl_fit.h>
#include <vector>

//#include "Randomize.hh"
//#include "G4Poisson.hh"
//#include "G4Step.hh"
//#include "G4Material.hh"
//#include "G4DynamicParticle.hh"
//#include "G4ParticleDefinition.hh"
using namespace std;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
// The constructor setups various constants pluc eloss parameters
// for silicon.
LandauFP420::LandauFP420()
    : minNumberInteractionsBohr(10.0),
      theBohrBeta2(50.0 * keV / proton_mass_c2),
      minLoss(0.000001 * eV),
      problim(0.01),
      alim(10.),
      nmaxCont1(4),
      nmaxCont2(16) {
  sumalim = -log(problim);

  chargeSquare = 1.;  // Assume all particles have charge 1
  // Taken from Geant4 printout, HARDWIRED for Silicon.
  ipotFluct = 0.0001736;        // material->GetIonisation()->GetMeanExcitationEnergy();
  electronDensity = 6.797E+20;  // material->GetElectronDensity();
  f1Fluct = 0.8571;             // material->GetIonisation()->GetF1fluct();
  f2Fluct = 0.1429;             // material->GetIonisation()->GetF2fluct();
  e1Fluct = 0.000116;           // material->GetIonisation()->GetEnergy1fluct();
  e2Fluct = 0.00196;            // material->GetIonisation()->GetEnergy2fluct();
  e1LogFluct = -9.063;          // material->GetIonisation()->GetLogEnergy1fluct();
  e2LogFluct = -6.235;          // material->GetIonisation()->GetLogEnergy2fluct();
  rateFluct = 0.4;              // material->GetIonisation()->GetRateionexcfluct();
  ipotLogFluct = -8.659;        // material->GetIonisation()->GetLogMeanExcEnergy();
  e0 = 1.E-5;                   // material->GetIonisation()->GetEnergy0fluct();
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
LandauFP420::~LandauFP420() {}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
// The main dedx fluctuation routine.
// Arguments: momentum in MeV/c, mass in MeV, delta ray cut (tmax) in
// MeV, silicon thickness in mm, mean eloss in MeV.
double LandauFP420::SampleFluctuations(
    const double momentum, const double mass, double &tmax, const double length, const double meanLoss) {
  //  calculate actual loss from the mean loss
  //  The model used to get the fluctuation is essentially the same
  // as in Glandz in Geant3.

  // shortcut for very very small loss
  if (meanLoss < minLoss)
    return meanLoss;

  // if(dp->GetDefinition() != particle) {
  particleMass = mass;  // dp->GetMass();
  // G4double q     = dp->GetCharge();
  // chargeSquare   = q*q;
  //}

  // double gam   = (dp->GetKineticEnergy())/particleMass + 1.0;
  // double gam2  = gam*gam;
  double gam2 = (momentum * momentum) / (particleMass * particleMass) + 1.0;
  double beta2 = 1.0 - 1.0 / gam2;

  // Validity range for delta electron cross section
  double loss, siga;
  // Gaussian fluctuation
  if (meanLoss >= minNumberInteractionsBohr * tmax || tmax <= ipotFluct * minNumberInteractionsBohr) {
    siga = (1.0 / beta2 - 0.5) * twopi_mc2_rcl2 * tmax * length * electronDensity * chargeSquare;
    siga = sqrt(siga);
    do {
      // loss = G4RandGauss::shoot(meanLoss,siga);
      loss = CLHEP::RandGaussQ::shoot(meanLoss, siga);
    } while (loss < 0. || loss > 2. * meanLoss);

    return loss;
  }

  // Non Gaussian fluctuation
  double suma, w1, w2, C, lossc, w;
  double a1, a2, a3;
  int p1, p2, p3;
  int nb;
  double corrfac, na, alfa, rfac, namean, sa, alfa1, ea, sea;
  double dp3;

  w1 = tmax / ipotFluct;
  w2 = log(2. * electron_mass_c2 * (gam2 - 1.0));

  C = meanLoss * (1. - rateFluct) / (w2 - ipotLogFluct - beta2);

  a1 = C * f1Fluct * (w2 - e1LogFluct - beta2) / e1Fluct;
  a2 = C * f2Fluct * (w2 - e2LogFluct - beta2) / e2Fluct;
  a3 = rateFluct * meanLoss * (tmax - ipotFluct) / (ipotFluct * tmax * log(w1));
  if (a1 < 0.)
    a1 = 0.;
  if (a2 < 0.)
    a2 = 0.;
  if (a3 < 0.)
    a3 = 0.;

  suma = a1 + a2 + a3;

  loss = 0.;

  if (suma < sumalim)  // very small Step
  {
    // e0 = material->GetIonisation()->GetEnergy0fluct();//Hardwired in const
    if (tmax == ipotFluct) {
      a3 = meanLoss / e0;

      if (a3 > alim) {
        siga = sqrt(a3);
        // p3 = G4std::max(0,int(G4RandGauss::shoot(a3,siga)+0.5));
        p3 = std::max(0, int(CLHEP::RandGaussQ::shoot(a3, siga) + 0.5));
      } else
        p3 = CLHEP::RandPoisson::shoot(a3);
      // p3 = G4Poisson(a3);

      loss = p3 * e0;

      if (p3 > 0)
        // loss += (1.-2.*G4UniformRand())*e0 ;
        loss += (1. - 2. * CLHEP::RandFlat::shoot()) * e0;

    } else {
      tmax = tmax - ipotFluct + e0;
      a3 = meanLoss * (tmax - e0) / (tmax * e0 * log(tmax / e0));

      if (a3 > alim) {
        siga = sqrt(a3);
        // p3 = G4std::max(0,int(G4RandGauss::shoot(a3,siga)+0.5));
        p3 = std::max(0, int(CLHEP::RandGaussQ::shoot(a3, siga) + 0.5));
      } else
        p3 = CLHEP::RandPoisson::shoot(a3);
      // p3 = G4Poisson(a3);

      if (p3 > 0) {
        w = (tmax - e0) / tmax;
        if (p3 > nmaxCont2) {
          dp3 = float(p3);
          corrfac = dp3 / float(nmaxCont2);
          p3 = nmaxCont2;
        } else
          corrfac = 1.;

        // for(int i=0; i<p3; i++) loss += 1./(1.-w*G4UniformRand()) ;
        for (int i = 0; i < p3; i++)
          loss += 1. / (1. - w * CLHEP::RandFlat::shoot());
        loss *= e0 * corrfac;
      }
    }
  }

  else  // not so small Step
  {
    // excitation type 1
    if (a1 > alim) {
      siga = sqrt(a1);
      // p1 = std::max(0,int(G4RandGauss::shoot(a1,siga)+0.5));
      p1 = std::max(0, int(CLHEP::RandGaussQ::shoot(a1, siga) + 0.5));
    } else
      p1 = CLHEP::RandPoisson::shoot(a1);
    // p1 = G4Poisson(a1);

    // excitation type 2
    if (a2 > alim) {
      siga = sqrt(a2);
      // p2 = std::max(0,int(G4RandGauss::shoot(a2,siga)+0.5));
      p2 = std::max(0, int(CLHEP::RandGaussQ::shoot(a2, siga) + 0.5));
    } else
      p2 = CLHEP::RandPoisson::shoot(a2);
    // p2 = G4Poisson(a2);

    loss = p1 * e1Fluct + p2 * e2Fluct;

    // smearing to avoid unphysical peaks
    if (p2 > 0)
      // loss += (1.-2.*G4UniformRand())*e2Fluct;
      loss += (1. - 2. * CLHEP::RandFlat::shoot()) * e2Fluct;
    else if (loss > 0.)
      loss += (1. - 2. * CLHEP::RandFlat::shoot()) * e1Fluct;

    // ionisation .......................................
    if (a3 > 0.) {
      if (a3 > alim) {
        siga = sqrt(a3);
        p3 = std::max(0, int(CLHEP::RandGaussQ::shoot(a3, siga) + 0.5));
      } else
        p3 = CLHEP::RandPoisson::shoot(a3);

      lossc = 0.;
      if (p3 > 0) {
        na = 0.;
        alfa = 1.;
        if (p3 > nmaxCont2) {
          dp3 = float(p3);
          rfac = dp3 / (float(nmaxCont2) + dp3);
          namean = float(p3) * rfac;
          sa = float(nmaxCont1) * rfac;
          na = CLHEP::RandGaussQ::shoot(namean, sa);
          if (na > 0.) {
            alfa = w1 * float(nmaxCont2 + p3) / (w1 * float(nmaxCont2) + float(p3));
            alfa1 = alfa * log(alfa) / (alfa - 1.);
            ea = na * ipotFluct * alfa1;
            sea = ipotFluct * sqrt(na * (alfa - alfa1 * alfa1));
            lossc += CLHEP::RandGaussQ::shoot(ea, sea);
          }
        }

        nb = int(float(p3) - na);
        if (nb > 0) {
          w2 = alfa * ipotFluct;
          w = (tmax - w2) / tmax;
          for (int k = 0; k < nb; k++)
            lossc += w2 / (1. - w * CLHEP::RandFlat::shoot());
        }
      }
      loss += lossc;
    }
  }

  return loss;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

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
// $Id: SiG4UniversalFluctuation.cc,v 1.9 2013/04/25 18:17:07 civanch Exp $
// GEANT4 tag $Name: CMSSW_6_2_0 $
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
// 16-10-03 Changed interface to Initialisation (V.Ivanchenko)
// 07-11-03 Fix problem of rounding of double in G4UniversalFluctuations
// 06-02-04 Add control on big sigma > 2*meanLoss (V.Ivanchenko)
// 26-04-04 Comment out the case of very small step (V.Ivanchenko)
// 07-02-05 define problim = 5.e-3 (mma)
// 03-05-05 conditions of Gaussian fluctuation changed (bugfix)
//          + smearing for very small loss (L.Urban)
//  
// Modified for standalone use in CMSSW. danek k. 2/06        
// 25-04-13 Used vdt::log, added check a3>0 (V.Ivanchenko & D. Nikolopoulos)         

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"
#include <math.h>
#include "vdt/log.h"
//#include "G4UniversalFluctuation.hh"
//#include "Randomize.hh"
//#include "G4Poisson.hh"
//#include "G4Step.hh"
//#include "G4Material.hh"
//#include "G4DynamicParticle.hh"
//#include "G4ParticleDefinition.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

using namespace std;

SiG4UniversalFluctuation::SiG4UniversalFluctuation(CLHEP::HepRandomEngine& eng)
  :rndEngine(eng),
   gaussQDistribution(0),
   poissonQDistribution(0),
   flatDistribution(0),
   minNumberInteractionsBohr(10.0),
   theBohrBeta2(50.0*keV/proton_mass_c2),
   minLoss(10.*eV),
   problim(5.e-3),  
   alim(10.),
   nmaxCont1(4.),
   nmaxCont2(16.)
{
  sumalim = -log(problim);
  //lastMaterial = 0;
  
  // Add these definitions d.k.
  chargeSquare   = 1.;  //Assume all particles have charge 1
  // Taken from Geant4 printout, HARDWIRED for Silicon.
  ipotFluct = 0.0001736; //material->GetIonisation()->GetMeanExcitationEnergy();  
  electronDensity = 6.797E+20; // material->GetElectronDensity();
  f1Fluct      = 0.8571; // material->GetIonisation()->GetF1fluct();
  f2Fluct      = 0.1429; //material->GetIonisation()->GetF2fluct();
  e1Fluct      = 0.000116;// material->GetIonisation()->GetEnergy1fluct();
  e2Fluct      = 0.00196; //material->GetIonisation()->GetEnergy2fluct();
  e1LogFluct   = -9.063;  //material->GetIonisation()->GetLogEnergy1fluct();
  e2LogFluct   = -6.235;  //material->GetIonisation()->GetLogEnergy2fluct();
  rateFluct    = 0.4;     //material->GetIonisation()->GetRateionexcfluct();
  ipotLogFluct = -8.659;  //material->GetIonisation()->GetLogMeanExcEnergy();
  e0 = 1.E-5;             //material->GetIonisation()->GetEnergy0fluct();
  
  gaussQDistribution = new CLHEP::RandGaussQ(rndEngine);
  poissonQDistribution = new CLHEP::RandPoissonQ(rndEngine);
  flatDistribution = new CLHEP::RandFlat(rndEngine);

  //cout << " init new fluct +++++++++++++++++++++++++++++++++++++++++"<<endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// The main dedx fluctuation routine.
// Arguments: momentum in MeV/c, mass in MeV, delta ray cut (tmax) in
// MeV, silicon thickness in mm, mean eloss in MeV.

SiG4UniversalFluctuation::~SiG4UniversalFluctuation()
{
  delete gaussQDistribution;
  delete poissonQDistribution;
  delete flatDistribution;

}


double SiG4UniversalFluctuation::SampleFluctuations(const double momentum,
						    const double mass,
						    double& tmax,
						    const double length,
						    const double meanLoss)
{
// Calculate actual loss from the mean loss.
// The model used to get the fluctuations is essentially the same
// as in Glandz in Geant3 (Cern program library W5013, phys332).
// L. Urban et al. NIM A362, p.416 (1995) and Geant4 Physics Reference Manual

  // shortcut for very very small loss (out of validity of the model)
  //
  if (meanLoss < minLoss) return meanLoss;

  //if(!particle) InitialiseMe(dp->GetDefinition());
  //G4double tau   = dp->GetKineticEnergy()/particleMass;
  //G4double gam   = tau + 1.0;
  //G4double gam2  = gam*gam;
  //G4double beta2 = tau*(tau + 2.0)/gam2;

  particleMass   = mass; // dp->GetMass();
  double gam2   = (momentum*momentum)/(particleMass*particleMass) + 1.0;
  double beta2 = 1.0 - 1.0/gam2;
  double gam = sqrt(gam2);

  double loss(0.), siga(0.);
  
  // Gaussian regime
  // for heavy particles only and conditions
  // for Gauusian fluct. has been changed 
  //
  if ((particleMass > electron_mass_c2) &&
      (meanLoss >= minNumberInteractionsBohr*tmax))
  {
    double massrate = electron_mass_c2/particleMass ;
    double tmaxkine = 2.*electron_mass_c2*beta2*gam2/
      (1.+massrate*(2.*gam+massrate)) ;
    if (tmaxkine <= 2.*tmax)   
    {
      //electronDensity = material->GetElectronDensity();
      siga  = (1.0/beta2 - 0.5) * twopi_mc2_rcl2 * tmax * length
                                * electronDensity * chargeSquare;
      siga = sqrt(siga);
      double twomeanLoss = meanLoss + meanLoss;
      if (twomeanLoss < siga) {
        double x;
        do {
          loss = twomeanLoss*flatDistribution->fire();
       	  x = (loss - meanLoss)/siga;
        } while (1.0 - 0.5*x*x < flatDistribution->fire());
      } else {
        do {
          loss = gaussQDistribution->fire(meanLoss,siga);
        } while (loss < 0. || loss > twomeanLoss);
      }
      return loss;
    }
  }

  // Glandz regime : initialisation
  //
//   if (material != lastMaterial) {
//     f1Fluct      = material->GetIonisation()->GetF1fluct();
//     f2Fluct      = material->GetIonisation()->GetF2fluct();
//     e1Fluct      = material->GetIonisation()->GetEnergy1fluct();
//     e2Fluct      = material->GetIonisation()->GetEnergy2fluct();
//     e1LogFluct   = material->GetIonisation()->GetLogEnergy1fluct();
//     e2LogFluct   = material->GetIonisation()->GetLogEnergy2fluct();
//     rateFluct    = material->GetIonisation()->GetRateionexcfluct();
//     ipotFluct    = material->GetIonisation()->GetMeanExcitationEnergy();
//     ipotLogFluct = material->GetIonisation()->GetLogMeanExcEnergy();
//     lastMaterial = material;
//   }

  double a1 = 0. , a2 = 0., a3 = 0. ;
  double p1,p2,p3;
  double rate = rateFluct ;

  double w1 = tmax/ipotFluct;
  double w2 = vdt::fast_log(2.*electron_mass_c2*beta2*gam2)-beta2;

  if(w2 > ipotLogFluct)
  {
    double C = meanLoss*(1.-rateFluct)/(w2-ipotLogFluct);
    a1 = C*f1Fluct*(w2-e1LogFluct)/e1Fluct;
    a2 = C*f2Fluct*(w2-e2LogFluct)/e2Fluct;
    if(a2 < 0.)
    {
      a1 = 0. ;
      a2 = 0. ;
      rate = 1. ;  
    }
  }
  else
  {
    rate = 1. ;
  }

  // added
  if(tmax > ipotFluct) {
    a3 = rate*meanLoss*(tmax-ipotFluct)/(ipotFluct*tmax*vdt::fast_log(w1));
  }
  double suma = a1+a2+a3;
  
  // Glandz regime
  //
  if (suma > sumalim)
  {
    p1 = 0., p2 = 0 ;
    if((a1+a2) > 0.)
    {
      // excitation type 1
      if (a1>alim) {
        siga=sqrt(a1) ;
        p1 = max(0.,gaussQDistribution->fire(a1,siga)+0.5);
      } else {
        p1 = double(poissonQDistribution->fire(a1));
      }
    
      // excitation type 2
      if (a2>alim) {
        siga=sqrt(a2) ;
        p2 = max(0.,gaussQDistribution->fire(a2,siga)+0.5);
      } else {
        p2 = double(poissonQDistribution->fire(a2));
      }
    
      loss = p1*e1Fluct+p2*e2Fluct;
 
      // smearing to avoid unphysical peaks
      if (p2 > 0.)
        loss += (1.-2.*flatDistribution->fire())*e2Fluct;
      else if (loss>0.)
        loss += (1.-2.*flatDistribution->fire())*e1Fluct;   
      if (loss < 0.) loss = 0.0;
    }

    // ionisation
    if (a3 > 0.) {
      if (a3>alim) {
        siga=sqrt(a3) ;
        p3 = max(0.,gaussQDistribution->fire(a3,siga)+0.5);
      } else {
        p3 = double(poissonQDistribution->fire(a3));
      }
      double lossc = 0.;
      if (p3 > 0) {
        double na = 0.; 
        double alfa = 1.;
        if (p3 > nmaxCont2) {
          double rfac   = p3/(nmaxCont2+p3);
          double namean = p3*rfac;
          double sa     = nmaxCont1*rfac;
          na              = gaussQDistribution->fire(namean,sa);
          if (na > 0.) {
            alfa   = w1*(nmaxCont2+p3)/(w1*nmaxCont2+p3);
            double alfa1  = alfa*vdt::fast_log(alfa)/(alfa-1.);
            double ea     = na*ipotFluct*alfa1;
            double sea    = ipotFluct*sqrt(na*(alfa-alfa1*alfa1));
            lossc += gaussQDistribution->fire(ea,sea);
          }
        }

        if (p3 > na) {
          w2 = alfa*ipotFluct;
          double w  = (tmax-w2)/tmax;
          int    nb = int(p3-na);
          for (int k=0; k<nb; k++) lossc += w2/(1.-w*flatDistribution->fire());
        }
      }        
      loss += lossc;  
    }
    return loss;
  }
  
  // suma < sumalim;  very small energy loss;  
  //
  //double e0 = material->GetIonisation()->GetEnergy0fluct();

  a3 = meanLoss*(tmax-e0)/(tmax*e0*vdt::fast_log(tmax/e0));
  if (a3 > alim)
  {
    siga=sqrt(a3);
    p3 = max(0.,gaussQDistribution->fire(a3,siga)+0.5);
  } else {
    p3 = double(poissonQDistribution->fire(a3));
  }
  if (p3 > 0.) {
    double w = (tmax-e0)/tmax;
    double corrfac = 1.;
    if (p3 > nmaxCont2) {
      corrfac = p3/nmaxCont2;
      p3 = nmaxCont2;
    } 
    int ip3 = (int)p3;
    for (int i=0; i<ip3; i++) loss += 1./(1.-w*flatDistribution->fire());
    loss *= e0*corrfac;
    // smearing for losses near to e0
    if(p3 <= 2.)
      loss += e0*(1.-2.*flatDistribution->fire()) ;
   }
    
   return loss;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// G4double SiG4UniversalFluctuation::Dispersion(
//                           const G4Material* material,
//                           const G4DynamicParticle* dp,
//  				G4double& tmax,
// 			        G4double& length)
// {
//   if(!particle) InitialiseMe(dp->GetDefinition());

//   electronDensity = material->GetElectronDensity();

//   G4double gam   = (dp->GetKineticEnergy())/particleMass + 1.0;
//   G4double beta2 = 1.0 - 1.0/(gam*gam);

//   G4double siga  = (1.0/beta2 - 0.5) * twopi_mc2_rcl2 * tmax * length
//                  * electronDensity * chargeSquare;

//   return siga;
// }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

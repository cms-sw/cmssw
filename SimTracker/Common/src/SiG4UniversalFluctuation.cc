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
// GEANT4 tag $Name:  $
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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

using namespace std;

SiG4UniversalFluctuation::SiG4UniversalFluctuation()
  :minNumberInteractionsBohr(10.0),
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
  
  //cout << " init new fluct +++++++++++++++++++++++++++++++++++++++++"<<endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// The main dedx fluctuation routine.
// Arguments: momentum in MeV/c, mass in MeV, delta ray cut (tmax) in
// MeV, silicon thickness in mm, mean eloss in MeV.

SiG4UniversalFluctuation::~SiG4UniversalFluctuation()
{
}


double SiG4UniversalFluctuation::SampleFluctuations(const double momentum,
						    const double mass,
						    double& tmax,
						    const double length,
						    const double meanLoss,
                                                    CLHEP::HepRandomEngine* engine)
{
// Calculate actual loss from the mean loss.
// The model used to get the fluctuations is essentially the same
// as in Glandz in Geant3 (Cern program library W5013, phys332).
// L. Urban et al. NIM A362, p.416 (1995) and Geant4 Physics Reference Manual

  // shortcut for very very small loss (out of validity of the model)
  //
  if (meanLoss < minLoss) return meanLoss;

  particleMass   = mass; 
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
          loss = twomeanLoss*CLHEP::RandFlat::shoot(engine);
       	  x = (loss - meanLoss)/siga;
        } while (1.0 - 0.5*x*x < CLHEP::RandFlat::shoot(engine));
      } else {
        do {
          loss = CLHEP::RandGaussQ::shoot(engine, meanLoss, siga);
        } while (loss < 0. || loss > twomeanLoss);
      }
      return loss;
    }
  }

  double a1 = 0., a2 = 0., a3 = 0.;
  double p3;
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
    if((a1+a2) > 0.)
    {
      double p1, p2;
      // excitation type 1
      if (a1>alim) {
        siga=sqrt(a1) ;
        p1 = max(0., CLHEP::RandGaussQ::shoot(engine, a1, siga) + 0.5);
      } else {
	p1 = double(CLHEP::RandPoissonQ::shoot(engine,a1));
      }
    
      // excitation type 2
      if (a2>alim) {
        siga=sqrt(a2) ;
        p2 = max(0., CLHEP::RandGaussQ::shoot(engine, a2, siga) + 0.5);
      } else {
	p2 = double(CLHEP::RandPoissonQ::shoot(engine,a2));
      }
    
      loss = p1*e1Fluct+p2*e2Fluct;
 
      // smearing to avoid unphysical peaks
      if (p2 > 0.)
        loss += (1.-2.*CLHEP::RandFlat::shoot(engine))*e2Fluct;
      else if (loss>0.)
        loss += (1.-2.*CLHEP::RandFlat::shoot(engine))*e1Fluct;
      if (loss < 0.) loss = 0.0;
    }

    // ionisation
    if (a3 > 0.) {
      if (a3>alim) {
        siga=sqrt(a3) ;
        p3 = max(0., CLHEP::RandGaussQ::shoot(engine, a3, siga) + 0.5);
      } else {
	p3 = double(CLHEP::RandPoissonQ::shoot(engine,a3));
      }
      double lossc = 0.;
      if (p3 > 0) {
        double na = 0.; 
        double alfa = 1.;
        if (p3 > nmaxCont2) {
          double rfac   = p3/(nmaxCont2+p3);
          double namean = p3*rfac;
          double sa     = nmaxCont1*rfac;
          na              = CLHEP::RandGaussQ::shoot(engine, namean, sa);
          if (na > 0.) {
            alfa   = w1*(nmaxCont2+p3)/(w1*nmaxCont2+p3);
            double alfa1  = alfa*vdt::fast_log(alfa)/(alfa-1.);
            double ea     = na*ipotFluct*alfa1;
            double sea    = ipotFluct*sqrt(na*(alfa-alfa1*alfa1));
            lossc += CLHEP::RandGaussQ::shoot(engine, ea, sea);
          }
        }

        if (p3 > na) {
          w2 = alfa*ipotFluct;
          double w  = (tmax-w2)/tmax;
          int    nb = int(p3-na);
          for (int k=0; k<nb; k++) lossc += w2/(1.-w*CLHEP::RandFlat::shoot(engine));
        }
      }        
      loss += lossc;  
    }
    return loss;
  }
  
  // suma < sumalim;  very small energy loss;  
  a3 = meanLoss*(tmax-e0)/(tmax*e0*vdt::fast_log(tmax/e0));
  if (a3 > alim)
  {
    siga=sqrt(a3);
    p3 = max(0., CLHEP::RandGaussQ::shoot(engine, a3, siga) + 0.5);
  } else {
    p3 = double(CLHEP::RandPoissonQ::shoot(engine,a3));
  }
  if (p3 > 0.) {
    double w = (tmax-e0)/tmax;
    double corrfac = 1.;
    if (p3 > nmaxCont2) {
      corrfac = p3/nmaxCont2;
      p3 = nmaxCont2;
    } 
    int ip3 = (int)p3;
    for (int i=0; i<ip3; i++) loss += 1./(1.-w*CLHEP::RandFlat::shoot(engine));
    loss *= e0*corrfac;
    // smearing for losses near to e0
    if(p3 <= 2.)
      loss += e0*(1.-2.*CLHEP::RandFlat::shoot(engine)) ;
   }
    
   return loss;
}


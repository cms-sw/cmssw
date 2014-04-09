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
//
// $Id: UrbanMscModel93.cc 69120 2013-04-18 13:41:13Z vnivanch $
//
// -------------------------------------------------------------------
//   
// GEANT4 Class file
//    
//
// File name:   UrbanMscModel93
//
// Author:      Laszlo Urban, V.Ivanchenko copy it from G4UrbanMscModel93
//                            geant4-09-06-ref-07a global tag
//
// Creation date: 21.08.2013 
//
// Modifications:
//

// Class Description:
//
// Implementation of the model of multiple scattering based on
// H.W.Lewis Phys Rev 78 (1950) 526 and others

// -------------------------------------------------------------------
// In its present form the model can be  used for simulation 
//   of the e-/e+, muon and charged hadron multiple scattering
//


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "SimG4Core/PhysicsLists/interface/UrbanMscModel93.h"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include "G4Electron.hh"
#include "G4LossTableManager.hh"
#include "G4ParticleChangeForMSC.hh"

#include "G4Poisson.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

using namespace std;

static const G4double kappa = 2.5;
static const G4double kappapl1 = 3.5;
static const G4double kappami1 = 1.5;

UrbanMscModel93::UrbanMscModel93(const G4String& nam)
  : G4VMscModel(nam)
{
  masslimite    = 0.6*MeV;
  lambdalimit   = 1.*mm;
  fr            = 0.02;
  taubig        = 8.0;
  tausmall      = 1.e-16;
  taulim        = 1.e-6;
  currentTau    = taulim;
  tlimitminfix  = 1.e-6*mm;            
  stepmin       = tlimitminfix;
  smallstep     = 1.e10;
  currentRange  = 0. ;
  rangeinit     = 0.;
  tlimit        = 1.e10*mm;
  tlimitmin     = 10.*tlimitminfix;            
  tgeom         = 1.e50*mm;
  geombig       = 1.e50*mm;
  geommin       = 1.e-3*mm;
  geomlimit     = geombig;
  presafety     = 0.*mm;
                          
  y             = 0.;

  Zold          = 0.;
  Zeff          = 1.;
  Z2            = 1.;                
  Z23           = 1.;                    
  lnZ           = 0.;
  coeffth1      = 0.;
  coeffth2      = 0.;
  coeffc1       = 0.;
  coeffc2       = 0.;
  scr1ini       = fine_structure_const*fine_structure_const*
                  electron_mass_c2*electron_mass_c2/(0.885*0.885*4.*pi);
  scr2ini       = 3.76*fine_structure_const*fine_structure_const;
  scr1          = 0.;
  scr2          = 0.;

  theta0max     = pi/6.;
  rellossmax    = 0.50;
  third         = 1./3.;
  particle      = 0;
  theManager    = G4LossTableManager::Instance(); 
  firstStep     = true; 
  inside        = false;  
  insideskin    = false;

  numlim        = 0.01;
  xsi           = 3.;
  ea            = G4Exp(-xsi);
  eaa           = 1.-ea ;

  skindepth = skin*stepmin;

  mass = proton_mass_c2;
  charge = ChargeSquare = 1.0;
  currentKinEnergy = currentRadLength = lambda0 = lambdaeff = tPathLength 
    = zPathLength = par1 = par2 = par3 = 0;

  currentMaterialIndex = -1;
  fParticleChange = 0;
  couple = 0;
  SetSampleZ(false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

UrbanMscModel93::~UrbanMscModel93()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void UrbanMscModel93::Initialise(const G4ParticleDefinition* p,
				   const G4DataVector&)
{
  skindepth = skin*stepmin;

  // set values of some data members
  SetParticle(p);

  if(p->GetPDGMass() > MeV) {
    G4cout << "### WARNING: UrbanMscModel93 model is used for " 
	   << p->GetParticleName() << " !!! " << G4endl;
    G4cout << "###          This model should be used only for e+-" 
	   << G4endl;
  }

  fParticleChange = GetParticleChangeForMSC(p);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::ComputeCrossSectionPerAtom( 
                             const G4ParticleDefinition* part,
                                   G4double KineticEnergy,
                                   G4double AtomicNumber,G4double,
				   G4double, G4double)
{
  static const G4double sigmafactor = 
    twopi*classic_electr_radius*classic_electr_radius;
  static const G4double epsfactor = 2.*electron_mass_c2*electron_mass_c2*
                            Bohr_radius*Bohr_radius/(hbarc*hbarc);
  static const G4double epsmin = 1.e-4 , epsmax = 1.e10;

  static const G4double Zdat[15] = { 4.,  6., 13., 20., 26., 29., 32., 38., 47.,
				     50., 56., 64., 74., 79., 82. };

  static const G4double Tdat[22] = { 100*eV,  200*eV,  400*eV,  700*eV,
				     1*keV,   2*keV,   4*keV,   7*keV,
				     10*keV,  20*keV,  40*keV,  70*keV,
				     100*keV, 200*keV, 400*keV, 700*keV,
				     1*MeV,   2*MeV,   4*MeV,   7*MeV,
				     10*MeV,  20*MeV};

  // corr. factors for e-/e+ lambda for T <= Tlim
  static const G4double celectron[15][22] =
          {{1.125,1.072,1.051,1.047,1.047,1.050,1.052,1.054,
            1.054,1.057,1.062,1.069,1.075,1.090,1.105,1.111,
            1.112,1.108,1.100,1.093,1.089,1.087            },
           {1.408,1.246,1.143,1.096,1.077,1.059,1.053,1.051,
            1.052,1.053,1.058,1.065,1.072,1.087,1.101,1.108,
            1.109,1.105,1.097,1.090,1.086,1.082            },
           {2.833,2.268,1.861,1.612,1.486,1.309,1.204,1.156,
            1.136,1.114,1.106,1.106,1.109,1.119,1.129,1.132,
            1.131,1.124,1.113,1.104,1.099,1.098            },
           {3.879,3.016,2.380,2.007,1.818,1.535,1.340,1.236,
            1.190,1.133,1.107,1.099,1.098,1.103,1.110,1.113,
            1.112,1.105,1.096,1.089,1.085,1.098            },
           {6.937,4.330,2.886,2.256,1.987,1.628,1.395,1.265,
            1.203,1.122,1.080,1.065,1.061,1.063,1.070,1.073,
            1.073,1.070,1.064,1.059,1.056,1.056            },
           {9.616,5.708,3.424,2.551,2.204,1.762,1.485,1.330,
            1.256,1.155,1.099,1.077,1.070,1.068,1.072,1.074,
            1.074,1.070,1.063,1.059,1.056,1.052            },
           {11.72,6.364,3.811,2.806,2.401,1.884,1.564,1.386,
            1.300,1.180,1.112,1.082,1.073,1.066,1.068,1.069,
            1.068,1.064,1.059,1.054,1.051,1.050            },
           {18.08,8.601,4.569,3.183,2.662,2.025,1.646,1.439,
            1.339,1.195,1.108,1.068,1.053,1.040,1.039,1.039,
            1.039,1.037,1.034,1.031,1.030,1.036            },
           {18.22,10.48,5.333,3.713,3.115,2.367,1.898,1.631,
            1.498,1.301,1.171,1.105,1.077,1.048,1.036,1.033,
            1.031,1.028,1.024,1.022,1.021,1.024            },
           {14.14,10.65,5.710,3.929,3.266,2.453,1.951,1.669,
            1.528,1.319,1.178,1.106,1.075,1.040,1.027,1.022,
            1.020,1.017,1.015,1.013,1.013,1.020            },
           {14.11,11.73,6.312,4.240,3.478,2.566,2.022,1.720,
            1.569,1.342,1.186,1.102,1.065,1.022,1.003,0.997,
            0.995,0.993,0.993,0.993,0.993,1.011            },
           {22.76,20.01,8.835,5.287,4.144,2.901,2.219,1.855,
            1.677,1.410,1.224,1.121,1.073,1.014,0.986,0.976,
            0.974,0.972,0.973,0.974,0.975,0.987            },
           {50.77,40.85,14.13,7.184,5.284,3.435,2.520,2.059,
            1.837,1.512,1.283,1.153,1.091,1.010,0.969,0.954,
            0.950,0.947,0.949,0.952,0.954,0.963            },
           {65.87,59.06,15.87,7.570,5.567,3.650,2.682,2.182,
            1.939,1.579,1.325,1.178,1.108,1.014,0.965,0.947,
            0.941,0.938,0.940,0.944,0.946,0.954            },
           {55.60,47.34,15.92,7.810,5.755,3.767,2.760,2.239,
            1.985,1.609,1.343,1.188,1.113,1.013,0.960,0.939,
            0.933,0.930,0.933,0.936,0.939,0.949            }};
	    
  static const G4double cpositron[15][22] = {
           {2.589,2.044,1.658,1.446,1.347,1.217,1.144,1.110,
            1.097,1.083,1.080,1.086,1.092,1.108,1.123,1.131,
            1.131,1.126,1.117,1.108,1.103,1.100            },
           {3.904,2.794,2.079,1.710,1.543,1.325,1.202,1.145,
            1.122,1.096,1.089,1.092,1.098,1.114,1.130,1.137,
            1.138,1.132,1.122,1.113,1.108,1.102            },
           {7.970,6.080,4.442,3.398,2.872,2.127,1.672,1.451,
            1.357,1.246,1.194,1.179,1.178,1.188,1.201,1.205,
            1.203,1.190,1.173,1.159,1.151,1.145            },
           {9.714,7.607,5.747,4.493,3.815,2.777,2.079,1.715,
            1.553,1.353,1.253,1.219,1.211,1.214,1.225,1.228,
            1.225,1.210,1.191,1.175,1.166,1.174            },
           {17.97,12.95,8.628,6.065,4.849,3.222,2.275,1.820,
            1.624,1.382,1.259,1.214,1.202,1.202,1.214,1.219,
            1.217,1.203,1.184,1.169,1.160,1.151            },
           {24.83,17.06,10.84,7.355,5.767,3.707,2.546,1.996,
            1.759,1.465,1.311,1.252,1.234,1.228,1.238,1.241,
            1.237,1.222,1.201,1.184,1.174,1.159            },
           {23.26,17.15,11.52,8.049,6.375,4.114,2.792,2.155,
            1.880,1.535,1.353,1.281,1.258,1.247,1.254,1.256,
            1.252,1.234,1.212,1.194,1.183,1.170            },
           {22.33,18.01,12.86,9.212,7.336,4.702,3.117,2.348,
            2.015,1.602,1.385,1.297,1.268,1.251,1.256,1.258,
            1.254,1.237,1.214,1.195,1.185,1.179            },
           {33.91,24.13,15.71,10.80,8.507,5.467,3.692,2.808,
            2.407,1.873,1.564,1.425,1.374,1.330,1.324,1.320,
            1.312,1.288,1.258,1.235,1.221,1.205            },
           {32.14,24.11,16.30,11.40,9.015,5.782,3.868,2.917,
            2.490,1.925,1.596,1.447,1.391,1.342,1.332,1.327,
            1.320,1.294,1.264,1.240,1.226,1.214            },
           {29.51,24.07,17.19,12.28,9.766,6.238,4.112,3.066,
            2.602,1.995,1.641,1.477,1.414,1.356,1.342,1.336,
            1.328,1.302,1.270,1.245,1.231,1.233            },
           {38.19,30.85,21.76,15.35,12.07,7.521,4.812,3.498,
            2.926,2.188,1.763,1.563,1.484,1.405,1.382,1.371,
            1.361,1.330,1.294,1.267,1.251,1.239            },
           {49.71,39.80,27.96,19.63,15.36,9.407,5.863,4.155,
            3.417,2.478,1.944,1.692,1.589,1.480,1.441,1.423,
            1.409,1.372,1.330,1.298,1.280,1.258            },
           {59.25,45.08,30.36,20.83,16.15,9.834,6.166,4.407,
            3.641,2.648,2.064,1.779,1.661,1.531,1.482,1.459,
            1.442,1.400,1.354,1.319,1.299,1.272            },
           {56.38,44.29,30.50,21.18,16.51,10.11,6.354,4.542,
            3.752,2.724,2.116,1.817,1.692,1.554,1.499,1.474,
            1.456,1.412,1.364,1.328,1.307,1.282            }};

  //data/corrections for T > Tlim  
  static const G4double Tlim = 10.*MeV;
  static const G4double beta2lim = Tlim*(Tlim+2.*electron_mass_c2)/
                      ((Tlim+electron_mass_c2)*(Tlim+electron_mass_c2));
  static const G4double bg2lim   = Tlim*(Tlim+2.*electron_mass_c2)/
                      (electron_mass_c2*electron_mass_c2);

  static const G4double sig0[15] = {
                     0.2672*barn,  0.5922*barn, 2.653*barn,  6.235*barn,
                      11.69*barn  , 13.24*barn  , 16.12*barn, 23.00*barn ,
		      35.13*barn  , 39.95*barn  , 50.85*barn, 67.19*barn ,
                      91.15*barn  , 104.4*barn  , 113.1*barn};
		      		       
  static const G4double hecorr[15] = {
                         120.70, 117.50, 105.00, 92.92, 79.23,  74.510,  68.29,
                          57.39,  41.97,  36.14, 24.53, 10.21,  -7.855, -16.84,
			 -22.30};

  G4double sigma;
  SetParticle(part);

  Z23 = pow(AtomicNumber,2./3.); 

  // correction if particle .ne. e-/e+
  // compute equivalent kinetic energy
  // lambda depends on p*beta ....

  G4double eKineticEnergy = KineticEnergy;

  if(mass > electron_mass_c2)
  {
     G4double TAU = KineticEnergy/mass ;
     G4double c = mass*TAU*(TAU+2.)/(electron_mass_c2*(TAU+1.)) ;
     G4double w = c-2. ;
     G4double tau = 0.5*(w+sqrt(w*w+4.*c)) ;
     eKineticEnergy = electron_mass_c2*tau ;
  }

  G4double eTotalEnergy = eKineticEnergy + electron_mass_c2 ;
  G4double beta2 = eKineticEnergy*(eTotalEnergy+electron_mass_c2)
                                 /(eTotalEnergy*eTotalEnergy);
  G4double bg2   = eKineticEnergy*(eTotalEnergy+electron_mass_c2)
                                 /(electron_mass_c2*electron_mass_c2);

  G4double eps = epsfactor*bg2/Z23;

  if     (eps<epsmin)  sigma = 2.*eps*eps;
  else if(eps<epsmax)  sigma = G4Log(1.+2.*eps)-2.*eps/(1.+2.*eps);
  else                 sigma = G4Log(2.*eps)-1.+1./eps;

  sigma *= ChargeSquare*AtomicNumber*AtomicNumber/(beta2*bg2);

  // interpolate in AtomicNumber and beta2 
  G4double c1,c2,cc1,cc2,corr;

  // get bin number in Z
  G4int iZ = 14;
  while ((iZ>=0)&&(Zdat[iZ]>=AtomicNumber)) iZ -= 1;
  if (iZ==14)                               iZ = 13;
  if (iZ==-1)                               iZ = 0 ;

  G4double ZZ1 = Zdat[iZ];
  G4double ZZ2 = Zdat[iZ+1];
  G4double ratZ = (AtomicNumber-ZZ1)*(AtomicNumber+ZZ1)/
                  ((ZZ2-ZZ1)*(ZZ2+ZZ1));

  if(eKineticEnergy <= Tlim) 
  {
    // get bin number in T (beta2)
    G4int iT = 21;
    while ((iT>=0)&&(Tdat[iT]>=eKineticEnergy)) iT -= 1;
    if(iT==21)                                  iT = 20;
    if(iT==-1)                                  iT = 0 ;

    //  calculate betasquare values
    G4double T = Tdat[iT],   E = T + electron_mass_c2;
    G4double b2small = T*(E+electron_mass_c2)/(E*E);

    T = Tdat[iT+1]; E = T + electron_mass_c2;
    G4double b2big = T*(E+electron_mass_c2)/(E*E);
    G4double ratb2 = (beta2-b2small)/(b2big-b2small);

    if (charge < 0.)
    {
       c1 = celectron[iZ][iT];
       c2 = celectron[iZ+1][iT];
       cc1 = c1+ratZ*(c2-c1);

       c1 = celectron[iZ][iT+1];
       c2 = celectron[iZ+1][iT+1];
       cc2 = c1+ratZ*(c2-c1);

       corr = cc1+ratb2*(cc2-cc1);

       sigma *= sigmafactor/corr;
    }
    else              
    {
       c1 = cpositron[iZ][iT];
       c2 = cpositron[iZ+1][iT];
       cc1 = c1+ratZ*(c2-c1);

       c1 = cpositron[iZ][iT+1];
       c2 = cpositron[iZ+1][iT+1];
       cc2 = c1+ratZ*(c2-c1);

       corr = cc1+ratb2*(cc2-cc1);

       sigma *= sigmafactor/corr;
    }
  }
  else
  {
    c1 = bg2lim*sig0[iZ]*(1.+hecorr[iZ]*(beta2-beta2lim))/bg2;
    c2 = bg2lim*sig0[iZ+1]*(1.+hecorr[iZ+1]*(beta2-beta2lim))/bg2;
    if((AtomicNumber >= ZZ1) && (AtomicNumber <= ZZ2))
      sigma = c1+ratZ*(c2-c1) ;
    else if(AtomicNumber < ZZ1)
      sigma = AtomicNumber*AtomicNumber*c1/(ZZ1*ZZ1);
    else if(AtomicNumber > ZZ2)
      sigma = AtomicNumber*AtomicNumber*c2/(ZZ2*ZZ2);
  }
  return sigma;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void UrbanMscModel93::StartTracking(G4Track* track)
{
  SetParticle(track->GetDynamicParticle()->GetDefinition());
  firstStep = true; 
  inside = false;
  insideskin = false;
  tlimit = geombig;
  stepmin = tlimitminfix ;
  tlimitmin = 10.*stepmin ;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::ComputeTruePathLengthLimit(
                             const G4Track& track,
			     G4double& currentMinimalStep)
{
  tPathLength = currentMinimalStep;
  const G4DynamicParticle* dp = track.GetDynamicParticle();
  G4StepPoint* sp = track.GetStep()->GetPreStepPoint();
  G4StepStatus stepStatus = sp->GetStepStatus();
  couple = track.GetMaterialCutsCouple();
  SetCurrentCouple(couple); 
  currentMaterialIndex = couple->GetIndex();
  currentKinEnergy = dp->GetKineticEnergy();
  currentRange = GetRange(particle,currentKinEnergy,couple);
  lambda0 = GetTransportMeanFreePath(particle,currentKinEnergy);

  // stop here if small range particle
  if(inside || tPathLength < tlimitminfix) { 
    return ConvertTrueToGeom(tPathLength, currentMinimalStep); 
  } 
  
  if(tPathLength > currentRange) { tPathLength = currentRange; }

  presafety = sp->GetSafety();

  // G4cout << "Urban2::StepLimit tPathLength= " 
  // 	 <<tPathLength<<" safety= " << presafety
  //        << " range= " <<currentRange<< " lambda= "<<lambda0
  // 	 << " Alg: " << steppingAlgorithm <<G4endl;

  // far from geometry boundary
  if(currentRange < presafety)
    {
      inside = true;
      return ConvertTrueToGeom(tPathLength, currentMinimalStep);  
    }

  // standard  version
  //
  if (steppingAlgorithm == fUseDistanceToBoundary)
    {
      //compute geomlimit and presafety 
      geomlimit = ComputeGeomLimit(track, presafety, currentRange);

      // is it far from boundary ?
      if(currentRange < presafety)
	{
	  inside = true;
	  return ConvertTrueToGeom(tPathLength, currentMinimalStep);   
	}

      smallstep += 1.;
      insideskin = false;

      if(firstStep || stepStatus == fGeomBoundary)
        {
          rangeinit = currentRange;
          if(firstStep) smallstep = 1.e10;
          else  smallstep = 1.;

          //define stepmin here (it depends on lambda!)
          //rough estimation of lambda_elastic/lambda_transport
          G4double rat = currentKinEnergy/MeV ;
          rat = 1.e-3/(rat*(10.+rat)) ;
          //stepmin ~ lambda_elastic
          stepmin = rat*lambda0;
          skindepth = skin*stepmin;
          //define tlimitmin
          tlimitmin = 10.*stepmin;
          if(tlimitmin < tlimitminfix) tlimitmin = tlimitminfix;
	  //G4cout << "rangeinit= " << rangeinit << " stepmin= " << stepmin
	  //	 << " tlimitmin= " << tlimitmin << " geomlimit= " << geomlimit <<G4endl;
          // constraint from the geometry
          if((geomlimit < geombig) && (geomlimit > geommin))
            {
              // geomlimit is a geometrical step length
              // transform it to true path length (estimation)
              if((1.-geomlimit/lambda0) > 0.)
                geomlimit = -lambda0*G4Log(1.-geomlimit/lambda0)+tlimitmin ;

              if(stepStatus == fGeomBoundary)
                tgeom = geomlimit/facgeom;
              else
                tgeom = 2.*geomlimit/facgeom;
            }
            else
              tgeom = geombig;
        }


      //step limit 
      tlimit = facrange*rangeinit;              
      if(tlimit < facsafety*presafety)
        tlimit = facsafety*presafety; 

      //lower limit for tlimit
      if(tlimit < tlimitmin) tlimit = tlimitmin;

      if(tlimit > tgeom) tlimit = tgeom;

      //G4cout << "tgeom= " << tgeom << " geomlimit= " << geomlimit  
      //      << " tlimit= " << tlimit << " presafety= " << presafety << G4endl;

      // shortcut
      if((tPathLength < tlimit) && (tPathLength < presafety) &&
         (smallstep >= skin) && (tPathLength < geomlimit-0.999*skindepth))
	return ConvertTrueToGeom(tPathLength, currentMinimalStep);   

      // step reduction near to boundary
      if(smallstep < skin)
	{
	  tlimit = stepmin;
	  insideskin = true;
	}
      else if(geomlimit < geombig)
	{
	  if(geomlimit > skindepth)
	    {
	      if(tlimit > geomlimit-0.999*skindepth)
		tlimit = geomlimit-0.999*skindepth;
	    }
	  else
	    {
	      insideskin = true;
	      if(tlimit > stepmin) tlimit = stepmin;
	    }
	}

      if(tlimit < stepmin) tlimit = stepmin;

      // randomize 1st step or 1st 'normal' step in volume
      if(firstStep || ((smallstep == skin) && !insideskin)) 
        { 
          G4double temptlimit = tlimit;
          if(temptlimit > tlimitmin)
          {
            do {
              temptlimit = G4RandGauss::shoot(tlimit,0.3*tlimit);        
               } while ((temptlimit < tlimitmin) || 
                        (temptlimit > 2.*tlimit-tlimitmin));
          }
          else
            temptlimit = tlimitmin;
          if(tPathLength > temptlimit) tPathLength = temptlimit;
        }
      else
        {  
          if(tPathLength > tlimit) tPathLength = tlimit  ; 
        }

    }
    // for 'normal' simulation with or without magnetic field 
    //  there no small step/single scattering at boundaries
  else if(steppingAlgorithm == fUseSafety)
    {
      // compute presafety again if presafety <= 0 and no boundary
      // i.e. when it is needed for optimization purposes
      if((stepStatus != fGeomBoundary) && (presafety < tlimitminfix)) 
	presafety = ComputeSafety(sp->GetPosition(),tPathLength); 

      // is far from boundary
      if(currentRange < presafety)
        {
          inside = true;
          return ConvertTrueToGeom(tPathLength, currentMinimalStep);  
        }

      if(firstStep || stepStatus == fGeomBoundary)
      {
        rangeinit = currentRange;
        fr = facrange;
        // 9.1 like stepping for e+/e- only (not for muons,hadrons)
        if(mass < masslimite) 
        {
          if(lambda0 > currentRange)
            rangeinit = lambda0;
          if(lambda0 > lambdalimit)
            fr *= 0.75+0.25*lambda0/lambdalimit;
        }

        //lower limit for tlimit
        G4double rat = currentKinEnergy/MeV ;
        rat = 1.e-3/(rat*(10.+rat)) ;
        tlimitmin = 10.*lambda0*rat;
        if(tlimitmin < tlimitminfix) tlimitmin = tlimitminfix;
      }
      //step limit
      tlimit = fr*rangeinit;               

      if(tlimit < facsafety*presafety)
        tlimit = facsafety*presafety;

      //lower limit for tlimit
      if(tlimit < tlimitmin) tlimit = tlimitmin;
      
      if(tPathLength > tlimit) tPathLength = tlimit;

    }
  
  // version similar to 7.1 (needed for some experiments)
  else
    {
      if (stepStatus == fGeomBoundary)
	{
	  if (currentRange > lambda0) tlimit = facrange*currentRange;
	  else                        tlimit = facrange*lambda0;

	  if(tlimit < tlimitmin) tlimit = tlimitmin;
	  if(tPathLength > tlimit) tPathLength = tlimit;
	}
    }
  //G4cout << "tPathLength= " << tPathLength 
  //	 << " currentMinimalStep= " << currentMinimalStep << G4endl;
  return ConvertTrueToGeom(tPathLength, currentMinimalStep);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::ComputeGeomPathLength(G4double)
{
  firstStep = false; 
  lambdaeff = lambda0;
  par1 = -1. ;  
  par2 = par3 = 0. ;  

  //  do the true -> geom transformation
  zPathLength = tPathLength;

  // z = t for very small tPathLength
  if(tPathLength < tlimitminfix) return zPathLength;

  // this correction needed to run MSC with eIoni and eBrem inactivated
  // and makes no harm for a normal run
  if(tPathLength > currentRange)
    tPathLength = currentRange ;

  G4double tau   = tPathLength/lambda0 ;

  if ((tau <= tausmall) || insideskin) {
    zPathLength  = tPathLength;
    if(zPathLength > lambda0) zPathLength = lambda0;
    return zPathLength;
  }

  G4double zmean = tPathLength;
  if (tPathLength < currentRange*dtrl) {
    if(tau < taulim) zmean = tPathLength*(1.-0.5*tau) ;
    else             zmean = lambda0*(1.-G4Exp(-tau));
  } else if(currentKinEnergy < mass || tPathLength == currentRange)  {
    par1 = 1./currentRange ;
    par2 = 1./(par1*lambda0) ;
    par3 = 1.+par2 ;
    if(tPathLength < currentRange)
      zmean = (1.-G4Exp(par3*G4Log(1.-tPathLength/currentRange)))/(par1*par3) ;
    else
      zmean = 1./(par1*par3) ;
  } else {
    G4double T1 = GetEnergy(particle,currentRange-tPathLength,couple);
    G4double lambda1 = GetTransportMeanFreePath(particle,T1);

    par1 = (lambda0-lambda1)/(lambda0*tPathLength) ;
    par2 = 1./(par1*lambda0) ;
    par3 = 1.+par2 ;
    zmean = (1.-G4Exp(par3*G4Log(lambda1/lambda0)))/(par1*par3) ;
  }

  zPathLength = zmean ;

  //  sample z
  if(samplez)
  {
    const G4double  ztmax = 0.99 ;
    G4double zt = zmean/tPathLength ;

    if (tPathLength > stepmin && zt < ztmax)              
    {
      G4double u,cz1;
      if(zt >= third)
      {
        G4double cz = 0.5*(3.*zt-1.)/(1.-zt) ;
        cz1 = 1.+cz ;
        G4double u0 = cz/cz1 ;
        G4double grej ;
        do {
            u = G4Exp(G4Log(G4UniformRand())/cz1) ;
            grej = G4Exp(cz*G4Log(u/u0))*(1.-u)/(1.-u0) ;
           } while (grej < G4UniformRand()) ;
      }
      else
      {
        cz1 = 1./zt-1.;
        u = 1.-G4Exp(G4Log(G4UniformRand())/cz1) ;
      }
      zPathLength = tPathLength*u ;
    }
  }

  if(zPathLength > lambda0) zPathLength = lambda0;
  //G4cout << "zPathLength= " << zPathLength << " lambda1= " << lambda0 << G4endl;
  return zPathLength;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::ComputeTrueStepLength(G4double geomStepLength)
{
  // step defined other than transportation 
  if(geomStepLength == zPathLength && tPathLength <= currentRange)
    return tPathLength;

  // t = z for very small step
  zPathLength = geomStepLength;
  tPathLength = geomStepLength;
  if(geomStepLength < tlimitminfix) return tPathLength;
  
  // recalculation
  if((geomStepLength > lambda0*tausmall) && !insideskin)
  {
    if(par1 <  0.)
      tPathLength = -lambda0*G4Log(1.-geomStepLength/lambda0) ;
    else 
    {
      if(par1*par3*geomStepLength < 1.)
        tPathLength = (1.-G4Exp(G4Log(1.-par1*par3*geomStepLength)/par3))/par1 ;
      else 
        tPathLength = currentRange;
    }  
  }
  if(tPathLength < geomStepLength) tPathLength = geomStepLength;

  //G4cout << "tPathLength= " << tPathLength << " step= " << geomStepLength << G4endl;

  return tPathLength;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ThreeVector& 
UrbanMscModel93::SampleScattering(const G4ThreeVector& oldDirection,
				    G4double safety)
{
  fDisplacement.set(0.0,0.0,0.0);
  G4double kineticEnergy = currentKinEnergy;
  if (tPathLength > currentRange*dtrl) {
    kineticEnergy = GetEnergy(particle,currentRange-tPathLength,couple);
  } else {
    kineticEnergy -= tPathLength*GetDEDX(particle,currentKinEnergy,couple);
  }
  if((kineticEnergy <= eV) || (tPathLength <= tlimitminfix) ||
     (tPathLength/tausmall < lambda0)) { return  fDisplacement; }

  G4double cth  = SampleCosineTheta(tPathLength,kineticEnergy);

  // protection against 'bad' cth values
  if(std::fabs(cth) > 1.) { return  fDisplacement; }

  // extra protection agaist high energy particles backscattered 
  //  if(cth < 1.0 - 1000*tPathLength/lambda0 && kineticEnergy > 20*MeV) { 
    //G4cout << "Warning: large scattering E(MeV)= " << kineticEnergy 
    //	   << " s(mm)= " << tPathLength/mm
    //	   << " 1-cosTheta= " << 1.0 - cth << G4endl;
    // do Gaussian central scattering
  //  if(kineticEnergy > 0.5*GeV && cth < 0.9) {
  /*
  if(cth < 1.0 - 1000*tPathLength/lambda0 && 
     cth < 0.9 && kineticEnergy > 500*MeV) { 
    G4ExceptionDescription ed;
    ed << particle->GetParticleName()
       << " E(MeV)= " << kineticEnergy/MeV
       << " Step(mm)= " << tPathLength/mm
       << " tau= " << tPathLength/lambda0
       << " in " << CurrentCouple()->GetMaterial()->GetName()
       << " CosTheta= " << cth 
       << " is too big";
    G4Exception("UrbanMscModel93::SampleScattering","em0004",
		JustWarning, ed,"");
  }
  */

  G4double sth  = sqrt((1.0 - cth)*(1.0 + cth));
  G4double phi  = twopi*G4UniformRand();
  G4double dirx = sth*cos(phi);
  G4double diry = sth*sin(phi);

  G4ThreeVector newDirection(dirx,diry,cth);
  newDirection.rotateUz(oldDirection);
  fParticleChange->ProposeMomentumDirection(newDirection);

  if (latDisplasment && safety > tlimitminfix) {

    G4double r = SampleDisplacement();
    /*    
    G4cout << "UrbanMscModel93::SampleSecondaries: e(MeV)= " << kineticEnergy
	   << " sinTheta= " << sth << " r(mm)= " << r
           << " trueStep(mm)= " << tPathLength
           << " geomStep(mm)= " << zPathLength
           << G4endl;
    */
    if(r > 0.)
      {
        G4double latcorr = LatCorrelation();
        if(latcorr > r) latcorr = r;

        // sample direction of lateral displacement
        // compute it from the lateral correlation
        G4double Phi = 0.;
        if(std::abs(r*sth) < latcorr)
          Phi  = twopi*G4UniformRand();
        else
        {
          G4double psi = std::acos(latcorr/(r*sth));
          if(G4UniformRand() < 0.5)
            Phi = phi+psi;
          else
            Phi = phi-psi;
        }

        dirx = r*std::cos(Phi);
        diry = r*std::sin(Phi);

	fDisplacement.set(dirx,diry,0.0);
	fDisplacement.rotateUz(oldDirection);
      }
  }
  return fDisplacement;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::SampleCosineTheta(G4double trueStepLength,
					    G4double KineticEnergy)
{
  G4double cth = 1. ;
  G4double tau = trueStepLength/lambda0 ;

  Zeff = couple->GetMaterial()->GetTotNbOfElectPerVolume()/
         couple->GetMaterial()->GetTotNbOfAtomsPerVolume() ;

  if(Zold != Zeff)  
    UpdateCache();

  if(insideskin)
  {
    //no scattering, single or plural scattering
    G4double mean = trueStepLength/stepmin ;

    G4int n = G4Poisson(mean);
    if(n > 0)
    {
      //screening (Moliere-Bethe)
      G4double mom2 = KineticEnergy*(2.*mass+KineticEnergy);
      G4double beta2 = mom2/((KineticEnergy+mass)*(KineticEnergy+mass));
      G4double ascr = scr1/mom2;
      ascr *= 1.13+scr2/beta2;
      G4double ascr1 = 1.+2.*ascr;
      G4double bp1=ascr1+1.;
      G4double bm1=ascr1-1.;

      // single scattering from screened Rutherford x-section
      G4double ct,st,phi;
      G4double sx=0.,sy=0.,sz=0.;
      for(G4int i=1; i<=n; i++)
      {
        ct = ascr1-bp1*bm1/(2.*G4UniformRand()+bm1);
        if(ct < -1.) ct = -1.;
        if(ct >  1.) ct =  1.; 
        st = sqrt(1.-ct*ct);
        phi = twopi*G4UniformRand();
        sx += st*cos(phi);
        sy += st*sin(phi);
        sz += ct;
      }
      cth = sz/sqrt(sx*sx+sy*sy+sz*sz);
    }
  }
  else
  {
    if(trueStepLength >= currentRange*dtrl) 
    {
      if(par1*trueStepLength < 1.)
	tau = -par2*G4Log(1.-par1*trueStepLength) ;
      // for the case if ioni/brems are inactivated
      // see the corresponding condition in ComputeGeomPathLength 
      else if(1.-KineticEnergy/currentKinEnergy > taulim)
	tau = taubig ;
    }
    currentTau = tau ;
    lambdaeff = trueStepLength/currentTau;
    currentRadLength = couple->GetMaterial()->GetRadlen();

    if (tau >= taubig) cth = -1.+2.*G4UniformRand();
    else if (tau >= tausmall)
    {
      G4double xmeanth, x2meanth;
      if(tau < numlim) {
	xmeanth = 1.0 - tau*(1.0 - 0.5*tau);
        x2meanth= 1.0 - tau*(5.0 - 6.25*tau)*third;
      } else {
	xmeanth = G4Exp(-tau);
	x2meanth = (1.+2.*G4Exp(-2.5*tau))*third;
      }
      G4double relloss = 1.-KineticEnergy/currentKinEnergy;

      if(relloss > rellossmax) 
        return SimpleScattering(xmeanth,x2meanth);

      G4double theta0 = ComputeTheta0(trueStepLength,KineticEnergy);

      //G4cout << "Theta0= " << theta0 << " theta0max= " << theta0max 
      //	     << "  sqrt(tausmall)= " << sqrt(tausmall) << G4endl;

      // protection for very small angles
      G4double theta2 = theta0*theta0;

      if(theta2 < tausmall) { return cth; }
    
      if(theta0 > theta0max) {
        return SimpleScattering(xmeanth,x2meanth);
      }

      G4double x = theta2*(1.0 - theta2/12.);
      if(theta2 > numlim) {
	G4double sth = 2.*sin(0.5*theta0);
	x = sth*sth;
      }

      G4double xmean1 = 1.-(1.-(1.+xsi)*ea)*x/eaa;
      G4double x0 = 1. - xsi*x;

      // G4cout << " xmean1= " << xmean1 << "  xmeanth= " << xmeanth << G4endl;

      if(xmean1 <= 0.999*xmeanth) {
        return SimpleScattering(xmeanth,x2meanth);
      }
      // from e- and muon scattering data                    
      G4double c = coeffc1+coeffc2*y; 

      // tail should not be too big
      if(c < 1.9) { 
	/*
	if(KineticEnergy > 200*MeV && c < 1.6) {
	  G4cout << "UrbanMscModel93::SampleCosineTheta: E(GeV)= " 
		 << KineticEnergy/GeV 
		 << " !!** c= " << c
		 << " **!! length(mm)= " << trueStepLength << " Zeff= " << Zeff 
		 << " " << couple->GetMaterial()->GetName()
		 << " tau= " << tau << G4endl;
	}
	*/
	c = 1.9; 
      }

      if(fabs(c-3.) < 0.001)      { c = 3.001; }
      else if(fabs(c-2.) < 0.001) { c = 2.001; }

      G4double c1 = c-1.;

      //from continuity of derivatives
      G4double b = 1.+(c-xsi)*x;

      G4double b1 = b+1.;
      G4double bx = c*x;

      G4double eb1 = pow(b1,c1);
      G4double ebx = pow(bx,c1);
      G4double d = ebx/eb1;

      // G4double xmean2 = (x0*eb1+ebx-(eb1*bx-b1*ebx)/(c-2.))/(eb1-ebx);
      G4double xmean2 = (x0 + d - (bx - b1*d)/(c-2.))/(1. - d);
      
      G4double f1x0 = ea/eaa;
      G4double f2x0 = c1/(c*(1. - d));
      G4double prob = f2x0/(f1x0+f2x0);

      G4double qprob = xmeanth/(prob*xmean1+(1.-prob)*xmean2);

      // sampling of costheta
      //G4cout << "c= " << c << " qprob= " << qprob << " eb1= " << eb1
      // << " c1= " << c1 << " b1= " << b1 << " bx= " << bx << " eb1= " << eb1
      //	     << G4endl;
      if(G4UniformRand() < qprob)
      {
        G4double var = 0;
        if(G4UniformRand() < prob) {
          cth = 1.+G4Log(ea+G4UniformRand()*eaa)*x;
        } else {
          var = (1.0 - d)*G4UniformRand();
          if(var < numlim*d) {
            var /= (d*c1); 
            cth = -1.0 + var*(1.0 - 0.5*var*c)*(2. + (c - xsi)*x);
	  } else {
	    cth = 1. + x*(c - xsi - c*pow(var + d, -1.0/c1));
	    //b-b1*bx/G4Exp(log(ebx+(eb1-ebx)*G4UniformRand())/c1) ;
	  }
	}
	if(KineticEnergy > 5*GeV && cth < 0.9) {
	  G4cout << "UrbanMscModel93::SampleCosineTheta: E(GeV)= " 
		 << KineticEnergy/GeV 
		 << " 1-cosT= " << 1 - cth
		 << " length(mm)= " << trueStepLength << " Zeff= " << Zeff 
		 << " tau= " << tau
		 << " prob= " << prob << " var= " << var << G4endl;
	  G4cout << "  c= " << c << " qprob= " << qprob << " eb1= " << eb1
		 << " ebx= " << ebx
		 << " c1= " << c1 << " b= " << b << " b1= " << b1 
		 << " bx= " << bx << " d= " << d
		 << " ea= " << ea << " eaa= " << eaa << G4endl;
	}
      }
      else {
        cth = -1.+2.*G4UniformRand();
	if(KineticEnergy > 5*GeV) {
	  G4cout << "UrbanMscModel93::SampleCosineTheta: E(GeV)= " 
		 << KineticEnergy/GeV 
		 << " length(mm)= " << trueStepLength << " Zeff= " << Zeff 
		 << " qprob= " << qprob << G4endl;
	}
      }
    }
  }  
  return cth ;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::SampleDisplacement()
{
  // Compute rmean = sqrt(<r**2>) from theory
  G4double rmean = 0.0;
  if ((currentTau >= tausmall) && !insideskin) {
    if (currentTau < taulim) {
      rmean = kappa*currentTau*currentTau*currentTau*
             (1.-kappapl1*currentTau*0.25)/6. ;

    } else {
      G4double etau = 0.0;
      if (currentTau<taubig) etau = G4Exp(-currentTau);
      rmean = -kappa*currentTau;
      rmean = -G4Exp(rmean)/(kappa*kappami1);
      rmean += currentTau-kappapl1/kappa+kappa*etau/kappami1;
    }
    if (rmean>0.) rmean = 2.*lambdaeff*sqrt(rmean*third);
    else          rmean = 0.;
  }

  if(rmean == 0.) return rmean;

  // protection against z > t ...........................
  G4double rmax = (tPathLength-zPathLength)*(tPathLength+zPathLength);
    if(rmax <= 0.)
      rmax = 0.;
    else
      rmax = sqrt(rmax);

  if(rmean >= rmax) return rmax;
     
  return rmean;
  // VI comment out for the time being
  /*
  //sample r (Gaussian distribution with a mean of rmean )
  G4double r = 0.;
  G4double sigma = min(rmean,rmax-rmean);
  sigma /= 3.;
  G4double rlow  = rmean-3.*sigma;
  G4double rhigh = rmean+3.*sigma;
  do {
      r = G4RandGauss::shoot(rmean,sigma);  
     } while ((r < rlow) || (r > rhigh));   

  return r;
  */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double UrbanMscModel93::LatCorrelation()
{
  G4double latcorr = 0.;
  if((currentTau >= tausmall) && !insideskin)
  {
    if(currentTau < taulim)
      latcorr = lambdaeff*kappa*currentTau*currentTau*
                (1.-kappapl1*currentTau*third)*third;
    else
    {
      G4double etau = 0.;
      if(currentTau < taubig) etau = G4Exp(-currentTau);
      latcorr = -kappa*currentTau;
      latcorr = G4Exp(latcorr)/kappami1;
      latcorr += 1.-kappa*etau/kappami1 ;
      latcorr *= 2.*lambdaeff*third ;
    }
  }

  return latcorr;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

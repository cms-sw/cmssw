// -------------------------------------------------------------------
//
//
// GEANT4 Class header file 
//
//
// File name: UrbanMscModel93
//
// Original author:    Laszlo Urban, 
//
//    V.Ivanchenko have copied from G4UrbanMscModel93 class
//                 of Geant4 global tag geant4-09-06-ref-07 
//                 and have adopted to CMSSW
//
// Creation date: 21.08.2013 
//
// Class Description:
//
// Implementation of the model of multiple scattering based on
// H.W.Lewis Phys Rev 78 (1950) 526 
// L.Urban CERN-OPEN-2006-077, Dec. 2006
// V.N.Ivanchenko et al., J.Phys: Conf. Ser. 219 (2010) 032045

// -------------------------------------------------------------------
// In its present form the model can be  used for simulation 
//   of the e-/e+, muon and charged hadron multiple scattering
// 
// This code was copied from Geant4 at the moment when it was removed 
// from Geant4 completly (together with G4UrbanMscModel91, 95, 96).
// Since that time Geant4 supports the unique class G4UrbanMscModel.
// It was shown in Geant4 internal validations that this last class
// provides more accurate simulation for various thin target tests.
// This main Geant4 model does is not provide exactly the same results 
// for CMS calorimeters run1 versus run2. To keep calorimeter response
// unchanged, CMS private version of the Urban model was created. It is
// basically the the same model used for run1 but it includes several 
// technical fixed introduced after run1. There fixes do not change
// results but allow to avoid numerical problems for very small steps
// and to improve a bit of the CPU performance.
//

#ifndef UrbanMscModel93_h
#define UrbanMscModel93_h 1

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "G4VMscModel.hh"
#include "G4MscStepLimitType.hh"
#include "G4Log.hh"
#include "G4Exp.hh"
#include "G4SystemOfUnits.hh"

class G4ParticleChangeForMSC;
class G4SafetyHelper;
class G4LossTableManager;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class UrbanMscModel93 : public G4VMscModel
{

public:

  UrbanMscModel93(const G4String& nam = "UrbanMsc93");

  ~UrbanMscModel93() override;

  void Initialise(const G4ParticleDefinition*, const G4DataVector&) override;

  void StartTracking(G4Track*) override;

  G4double ComputeCrossSectionPerAtom(const G4ParticleDefinition* particle,
				      G4double KineticEnergy,
				      G4double AtomicNumber,
				      G4double AtomicWeight=0., 
				      G4double cut =0.,
				      G4double emax=DBL_MAX) override;

  G4ThreeVector& SampleScattering(const G4ThreeVector&, G4double safety) override;

  G4double ComputeTruePathLengthLimit(const G4Track& track,
				      G4double& currentMinimalStep) override;

  G4double ComputeGeomPathLength(G4double truePathLength) override;

  G4double ComputeTrueStepLength(G4double geomStepLength) override;

  inline G4double ComputeTheta0(G4double truePathLength,
				G4double KineticEnergy);

private:

  G4double SampleCosineTheta(G4double trueStepLength, G4double KineticEnergy);

  G4double SampleDisplacement();

  G4double LatCorrelation();

  inline void SetParticle(const G4ParticleDefinition*);

  inline void UpdateCache();

  inline G4double SimpleScattering(G4double xmeanth, G4double x2meanth);

  //  hide assignment operator
  UrbanMscModel93 & operator=(const  UrbanMscModel93 &right) = delete;
  UrbanMscModel93(const  UrbanMscModel93&) = delete;

  const G4ParticleDefinition* particle;
  G4ParticleChangeForMSC*     fParticleChange;

  const G4MaterialCutsCouple* couple;
  G4LossTableManager*         theManager;

  G4double mass;
  G4double charge,ChargeSquare;
  G4double masslimite,lambdalimit,fr;

  G4double taubig;
  G4double tausmall;
  G4double taulim;
  G4double currentTau;
  G4double tlimit;
  G4double tlimitmin;
  G4double tlimitminfix;
  G4double tgeom;

  G4double geombig;
  G4double geommin;
  G4double geomlimit;
  G4double skindepth;
  G4double smallstep;

  G4double presafety;

  G4double lambda0;
  G4double lambdaeff;
  G4double tPathLength;
  G4double zPathLength;
  G4double par1,par2,par3;

  G4double stepmin;

  G4double currentKinEnergy;
  G4double currentRange; 
  G4double rangeinit;
  G4double currentRadLength;

  G4double numlim, xsi, ea, eaa;

  G4double theta0max,rellossmax;
  G4double third;

  G4int    currentMaterialIndex;

  G4double y;
  G4double Zold;
  G4double Zeff,Z2,Z23,lnZ;
  G4double coeffth1,coeffth2;
  G4double coeffc1,coeffc2;
  G4double scr1ini,scr2ini,scr1,scr2;

  G4bool   firstStep;
  G4bool   inside;
  G4bool   insideskin;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
void UrbanMscModel93::SetParticle(const G4ParticleDefinition* p)
{
  if (p != particle) {
    particle = p;
    mass = p->GetPDGMass();
    charge = p->GetPDGCharge()/CLHEP::eplus;
    ChargeSquare = charge*charge;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
void UrbanMscModel93::UpdateCache()                                   
{
  lnZ = G4Log(Zeff);
  //new correction in theta0 formula
  coeffth1 = (1.-8.7780e-2/Zeff)*(0.87+0.03*lnZ);                   
  coeffth2 = (4.0780e-2+1.7315e-4*Zeff)*(0.87+0.03*lnZ);              
  // tail parameters
  G4double lnZ1 = G4Log(Zeff+1.);
  coeffc1  = 2.943-0.197*lnZ1;                  
  coeffc2  = 0.0987-0.0143*lnZ1;                              
  // for single scattering
  Z2   = Zeff*Zeff;
  Z23  = G4Exp(2.*lnZ/3.);
  scr1 = scr1ini*Z23;
  scr2 = scr2ini*Z2*ChargeSquare;
                                              
  Zold = Zeff;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
G4double UrbanMscModel93::ComputeTheta0(G4double trueStepLength,
					G4double KineticEnergy)
{
  // for all particles take the width of the central part
  //  from a  parametrization similar to the Highland formula
  // ( Highland formula: Particle Physics Booklet, July 2002, eq. 26.10)
  static const G4double c_highland = 13.6*CLHEP::MeV;
  G4double invbetacp = sqrt((currentKinEnergy+mass)*(KineticEnergy+mass)/
			    (currentKinEnergy*(currentKinEnergy+2.*mass)*
			     KineticEnergy*(KineticEnergy+2.*mass)));
  y = trueStepLength/currentRadLength;
  G4double theta0 = c_highland*std::abs(charge)*sqrt(y)*invbetacp;
  // correction factor from e- scattering data
  theta0 *= (coeffth1+coeffth2*G4Log(y));             

  return theta0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline
G4double UrbanMscModel93::SimpleScattering(G4double xmeanth, G4double x2meanth)
{
  // 'large angle scattering'
  // 2 model functions with correct xmean and x2mean
  G4double a = (2.*xmeanth+9.*x2meanth-3.)/(2.*xmeanth-3.*x2meanth+1.);
  G4double prob = (a+2.)*xmeanth/a;

  // sampling
  G4double cth = 1.;
  if(G4UniformRand() < prob) {
    cth = -1.+2.*G4Exp(G4Log(G4UniformRand())/(a+1.));
  } else {
    cth = -1.+2.*G4UniformRand();
  }
  return cth;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif


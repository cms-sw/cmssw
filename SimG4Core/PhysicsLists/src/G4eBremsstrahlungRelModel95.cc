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
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     G4eBremsstrahlungRelModel95
//
// Author:        Andreas Schaelicke 
//
// Creation date: 12.08.2008
//
// Modifications:
//
// 13.11.08    add SetLPMflag and SetLPMconstant methods
// 13.11.08    change default LPMconstant value
// 13.10.10    add angular distributon interface (VI)
//
// Main References:
//  Y.-S.Tsai, Rev. Mod. Phys. 46 (1974) 815; Rev. Mod. Phys. 49 (1977) 421. 
//  S.Klein,  Rev. Mod. Phys. 71 (1999) 1501.
//  T.Stanev et.al., Phys. Rev. D25 (1982) 1291.
//  M.L.Ter-Mikaelian, High-energy Electromagnetic Processes in Condensed Media, Wiley, 1972.
//
// -------------------------------------------------------------------
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "G4eBremsstrahlungRelModel95.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "Randomize.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4ProductionCutsTable.hh"
#include "G4ParticleChangeForLoss.hh"
#include "G4LossTableManager.hh"
#include "G4ModifiedTsai.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

const G4double G4eBremsstrahlungRelModel95::xgi[]={ 0.0199, 0.1017, 0.2372, 0.4083,
						  0.5917, 0.7628, 0.8983, 0.9801 };
const G4double G4eBremsstrahlungRelModel95::wgi[]={ 0.0506, 0.1112, 0.1569, 0.1813,
					    0.1813, 0.1569, 0.1112, 0.0506 };
const G4double G4eBremsstrahlungRelModel95::Fel_light[]  = {0., 5.31  , 4.79  , 4.74 ,  4.71} ;
const G4double G4eBremsstrahlungRelModel95::Finel_light[] = {0., 6.144 , 5.621 , 5.805 , 5.924} ;

using namespace std;

G4eBremsstrahlungRelModel95::G4eBremsstrahlungRelModel95(const G4ParticleDefinition* p,
						     const G4String& name)
  : G4VEmModel(name),
    particle(0),
    bremFactor(fine_structure_const*classic_electr_radius*classic_electr_radius*16./3.),
    isElectron(true),
    fMigdalConstant(classic_electr_radius*electron_Compton_length*electron_Compton_length*4.0*pi),
    fLPMconstant(fine_structure_const*electron_mass_c2*electron_mass_c2/(4.*pi*hbarc)*0.5),
    fXiLPM(0), fPhiLPM(0), fGLPM(0),
    use_completescreening(true),isInitialised(false)
{
  fParticleChange = 0;
  theGamma = G4Gamma::Gamma();

  lowKinEnergy = 0.1*GeV;
  SetLowEnergyLimit(lowKinEnergy);  

  nist = G4NistManager::Instance();  

  SetLPMFlag(true);
  SetAngularDistribution(new G4ModifiedTsai());

  particleMass = kinEnergy = totalEnergy = currentZ = z13 = z23 = lnZ = Fel 
    = Finel = fCoulomb = fMax = densityFactor = densityCorr = lpmEnergy 
    = xiLPM = phiLPM = gLPM = klpm = kp = 0.0;

  energyThresholdLPM = 1.e39;

  InitialiseConstants();
  if(p) { SetParticle(p); }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4eBremsstrahlungRelModel95::InitialiseConstants()
{
  facFel = log(184.15);
  facFinel = log(1194.);

  preS1 = 1./(184.15*184.15);
  logTwo = log(2.);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4eBremsstrahlungRelModel95::~G4eBremsstrahlungRelModel95()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4eBremsstrahlungRelModel95::SetParticle(const G4ParticleDefinition* p)
{
  particle = p;
  particleMass = p->GetPDGMass();
  if(p == G4Electron::Electron()) { isElectron = true; }
  else                            { isElectron = false;}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4eBremsstrahlungRelModel95::SetupForMaterial(const G4ParticleDefinition*,
						 const G4Material* mat, 
						 G4double kineticEnergy)
{
  densityFactor = mat->GetElectronDensity()*fMigdalConstant;
  lpmEnergy = mat->GetRadlen()*fLPMconstant;

  // Threshold for LPM effect (i.e. below which LPM hidden by density effect) 
  if (LPMFlag()) {
    energyThresholdLPM=sqrt(densityFactor)*lpmEnergy;
  } else {
     energyThresholdLPM=1.e39;   // i.e. do not use LPM effect
  }
  // calculate threshold for density effect
  kinEnergy   = kineticEnergy;
  totalEnergy = kineticEnergy + particleMass;
  densityCorr = densityFactor*totalEnergy*totalEnergy;

  // define critical gamma energies (important for integration/dicing)
  klpm=totalEnergy*totalEnergy/lpmEnergy;
  kp=sqrt(densityCorr);
    
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4eBremsstrahlungRelModel95::Initialise(const G4ParticleDefinition* p,
					   const G4DataVector& cuts)
{
  if(p) { SetParticle(p); }

  lowKinEnergy  = LowEnergyLimit();

  currentZ = 0.;

  InitialiseElementSelectors(p, cuts);

  if(isInitialised) { return; }
  fParticleChange = GetParticleChangeForLoss();
  isInitialised = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4eBremsstrahlungRelModel95::ComputeDEDXPerVolume(
					     const G4Material* material,
                                             const G4ParticleDefinition* p,
                                                   G4double kineticEnergy,
                                                   G4double cutEnergy)
{
  if(!particle) { SetParticle(p); }
  if(kineticEnergy < lowKinEnergy) { return 0.0; }
  G4double cut = std::min(cutEnergy, kineticEnergy);
  if(cut == 0.0) { return 0.0; }

  SetupForMaterial(particle, material,kineticEnergy);

  const G4ElementVector* theElementVector = material->GetElementVector();
  const G4double* theAtomicNumDensityVector = material->GetAtomicNumDensityVector();

  G4double dedx = 0.0;

  //  loop for elements in the material
  for (size_t i=0; i<material->GetNumberOfElements(); i++) {

    G4VEmModel::SetCurrentElement((*theElementVector)[i]);
    SetCurrentElement((*theElementVector)[i]->GetZ());

    dedx += theAtomicNumDensityVector[i]*currentZ*currentZ*ComputeBremLoss(cut);
  }
  dedx *= bremFactor;


  return dedx;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4eBremsstrahlungRelModel95::ComputeBremLoss(G4double cut)
{
  G4double loss = 0.0;

  // number of intervals and integration step 
  G4double vcut = cut/totalEnergy;
  G4int n = (G4int)(20*vcut) + 3;
  G4double delta = vcut/G4double(n);

  G4double e0 = 0.0;
  G4double xs; 

  // integration
  for(G4int l=0; l<n; l++) {

    for(G4int i=0; i<8; i++) {

      G4double eg = (e0 + xgi[i]*delta)*totalEnergy;

      if(totalEnergy > energyThresholdLPM) {
	xs = ComputeRelDXSectionPerAtom(eg);
      } else {
	xs = ComputeDXSectionPerAtom(eg);
      }
      loss += wgi[i]*xs/(1.0 + densityCorr/(eg*eg));
    }
    e0 += delta;
  }

  loss *= delta*totalEnergy;

  return loss;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4eBremsstrahlungRelModel95::ComputeCrossSectionPerAtom(
                                              const G4ParticleDefinition* p,
					      G4double kineticEnergy, 
					      G4double Z,   G4double,
					      G4double cutEnergy, 
					      G4double maxEnergy)
{
  if(!particle) { SetParticle(p); }
  if(kineticEnergy < lowKinEnergy) { return 0.0; }
  G4double cut  = std::min(cutEnergy, kineticEnergy);
  G4double tmax = std::min(maxEnergy, kineticEnergy);

  if(cut >= tmax) { return 0.0; }

  SetCurrentElement(Z);

  G4double cross = ComputeXSectionPerAtom(cut);

  // allow partial integration
  if(tmax < kinEnergy) { cross -= ComputeXSectionPerAtom(tmax); }
  
  cross *= Z*Z*bremFactor;

  return cross;
}
 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


G4double G4eBremsstrahlungRelModel95::ComputeXSectionPerAtom(G4double cut)
{
  G4double cross = 0.0;

  // number of intervals and integration step 
  G4double vcut = log(cut/totalEnergy);
  G4double vmax = log(kinEnergy/totalEnergy);
  G4int n = (G4int)(0.45*(vmax - vcut)) + 4;
  //  n=1; //  integration test 
  G4double delta = (vmax - vcut)/G4double(n);

  G4double e0 = vcut;
  G4double xs; 

  // integration
  for(G4int l=0; l<n; l++) {

    for(G4int i=0; i<8; i++) {

      G4double eg = exp(e0 + xgi[i]*delta)*totalEnergy;

      if(totalEnergy > energyThresholdLPM) {
	xs = ComputeRelDXSectionPerAtom(eg);
      } else {
	xs = ComputeDXSectionPerAtom(eg);
      }
      cross += wgi[i]*xs/(1.0 + densityCorr/(eg*eg));
    }
    e0 += delta;
  }

  cross *= delta;

  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
void  G4eBremsstrahlungRelModel95::CalcLPMFunctions(G4double k)
{
  // *** calculate lpm variable s & sprime ***
  // Klein eqs. (78) & (79)
  G4double sprime = sqrt(0.125*k*lpmEnergy/(totalEnergy*(totalEnergy-k)));

  G4double s1 = preS1*z23;
  G4double logS1 = 2./3.*lnZ-2.*facFel;
  G4double logTS1 = logTwo+logS1;

  xiLPM = 2.;

  if (sprime>1) 
    xiLPM = 1.;
  else if (sprime>sqrt(2.)*s1) {
    G4double h  = log(sprime)/logTS1;
    xiLPM = 1+h-0.08*(1-h)*(1-sqr(1-h))/logTS1;
  }

  G4double s = sprime/sqrt(xiLPM); 

  // *** merging with density effect***  should be only necessary in region "close to" kp, e.g. k<100*kp
  // using Ter-Mikaelian eq. (20.9)
  G4double k2 = k*k;
  s = s * (1 + (densityCorr/k2) );

  // recalculate Xi using modified s above
  // Klein eq. (75)
  xiLPM = 1.;
  if (s<=s1) xiLPM = 2.;
  else if ( (s1<s) && (s<=1) ) xiLPM = 1. + log(s)/logS1;
  

  // *** calculate supression functions phi and G ***
  // Klein eqs. (77)
  G4double s2=s*s;
  G4double s3=s*s2;
  G4double s4=s2*s2;

  if (s<0.1) {
    // high suppression limit
    phiLPM = 6.*s - 18.84955592153876*s2 + 39.47841760435743*s3 
      - 57.69873135166053*s4;
    gLPM = 37.69911184307752*s2 - 236.8705056261446*s3 + 807.7822389*s4;
  }
  else if (s<1.9516) {
    // intermediate suppression
    // using eq.77 approxim. valid s<2.      
    phiLPM = 1.-exp(-6.*s*(1.+(3.-pi)*s)
		+s3/(0.623+0.795*s+0.658*s2));
    if (s<0.415827397755) {
      // using eq.77 approxim. valid 0.07<s<2
      G4double psiLPM = 1-exp(-4*s-8*s2/(1+3.936*s+4.97*s2-0.05*s3+7.50*s4));
      gLPM = 3*psiLPM-2*phiLPM;
    }
    else {
      // using alternative parametrisiation
      G4double pre = -0.16072300849123999 + s*3.7550300067531581 + s2*-1.7981383069010097 
	+ s3*0.67282686077812381 + s4*-0.1207722909879257;
      gLPM = tanh(pre);
    }
  }
  else {
    // low suppression limit valid s>2.
    phiLPM = 1. - 0.0119048/s4;
    gLPM = 1. - 0.0230655/s4;
  }

  // *** make sure suppression is smaller than 1 ***
  // *** caused by Migdal approximation in xi    ***
  if (xiLPM*phiLPM>1. || s>0.57)  { xiLPM=1./phiLPM; }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


G4double G4eBremsstrahlungRelModel95::ComputeRelDXSectionPerAtom(G4double gammaEnergy)
// Ultra relativistic model
//   only valid for very high energies, but includes LPM suppression
//    * complete screening
{
  if(gammaEnergy < 0.0) { return 0.0; }

  G4double y = gammaEnergy/totalEnergy;
  G4double y2 = y*y*.25;
  G4double yone2 = (1.-y+2.*y2);

  // ** form factors complete screening case **      

  // ** calc LPM functions -- include ter-mikaelian merging with density effect **
  //  G4double xiLPM, gLPM, phiLPM;  // to be made member variables !!!
  CalcLPMFunctions(gammaEnergy);

  G4double mainLPM   = xiLPM*(y2 * gLPM + yone2*phiLPM) * ( (Fel-fCoulomb) + Finel/currentZ );
  G4double secondTerm = (1.-y)/12.*(1.+1./currentZ);

  G4double cross = mainLPM+secondTerm;
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4eBremsstrahlungRelModel95::ComputeDXSectionPerAtom(G4double gammaEnergy)
// Relativistic model
//  only valid for high energies (and if LPM suppression does not play a role)
//  * screening according to thomas-fermi-Model (only valid for Z>5)
//  * no LPM effect
{

  if(gammaEnergy < 0.0) { return 0.0; }

  G4double y = gammaEnergy/totalEnergy;

  G4double main=0.,secondTerm=0.;

  if (use_completescreening|| currentZ<5) {
    // ** form factors complete screening case **      
    main   = (3./4.*y*y - y + 1.) * ( (Fel-fCoulomb) + Finel/currentZ );
    secondTerm = (1.-y)/12.*(1.+1./currentZ);
  }
  else {
    // ** intermediate screening using Thomas-Fermi FF from Tsai only valid for Z>=5** 
    G4double dd=100.*electron_mass_c2*y/(totalEnergy-gammaEnergy);
    G4double gg=dd*z13;
    G4double eps=dd*z23;
    G4double phi1=Phi1(gg,currentZ),  phi1m2=Phi1M2(gg,currentZ);
    G4double psi1=Psi1(eps,currentZ),  psi1m2=Psi1M2(eps,currentZ);
    
    main   = (3./4.*y*y - y + 1.) * ( (0.25*phi1-1./3.*lnZ-fCoulomb) + (0.25*psi1-2./3.*lnZ)/currentZ );
    secondTerm = (1.-y)/8.*(phi1m2+psi1m2/currentZ);
  }
  G4double cross = main+secondTerm;
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4eBremsstrahlungRelModel95::SampleSecondaries(
				      std::vector<G4DynamicParticle*>* vdp, 
				      const G4MaterialCutsCouple* couple,
				      const G4DynamicParticle* dp,
				      G4double cutEnergy,
				      G4double maxEnergy)
{
  G4double kineticEnergy = dp->GetKineticEnergy();
  if(kineticEnergy < lowKinEnergy) { return; }
  G4double cut  = std::min(cutEnergy, kineticEnergy);
  G4double emax = std::min(maxEnergy, kineticEnergy);
  if(cut >= emax) { return; }

  SetupForMaterial(particle, couple->GetMaterial(), kineticEnergy);

  const G4Element* elm = 
    SelectRandomAtom(couple,particle,kineticEnergy,cut,emax);
  SetCurrentElement(elm->GetZ());

  kinEnergy   = kineticEnergy;
  totalEnergy = kineticEnergy + particleMass;
  densityCorr = densityFactor*totalEnergy*totalEnergy;
  G4ThreeVector direction = dp->GetMomentumDirection();

  //G4double fmax= fMax;
  G4bool highe = true;
  if(totalEnergy < energyThresholdLPM) { highe = false; }
 
  G4double xmin = log(cut*cut + densityCorr);
  G4double xmax = log(emax*emax  + densityCorr);
  G4double gammaEnergy, f, x; 

  do {
    x = exp(xmin + G4UniformRand()*(xmax - xmin)) - densityCorr;
    if(x < 0.0) { x = 0.0; }
    gammaEnergy = sqrt(x);
    if(highe) { f = ComputeRelDXSectionPerAtom(gammaEnergy); }
    else      { f = ComputeDXSectionPerAtom(gammaEnergy); }

    if ( f > fMax ) {
      G4cout << "### G4eBremsstrahlungRelModel95 Warning: Majoranta exceeded! "
	     << f << " > " << fMax
	     << " Egamma(MeV)= " << gammaEnergy
	     << " Ee(MeV)= " << kineticEnergy
	     << "  " << GetName()
	     << G4endl;
    }

  } while (f < fMax*G4UniformRand());

  //
  // angles of the emitted gamma. ( Z - axis along the parent particle)
  // use general interface
  //
  G4double theta = GetAngularDistribution()->PolarAngle(totalEnergy,
							totalEnergy-gammaEnergy,
							(G4int)currentZ);

  G4double sint = sin(theta);
  G4double phi = twopi * G4UniformRand();
  G4ThreeVector gammaDirection(sint*cos(phi),sint*sin(phi), cos(theta));
  gammaDirection.rotateUz(direction);

  // create G4DynamicParticle object for the Gamma
  G4DynamicParticle* g = new G4DynamicParticle(theGamma,gammaDirection,
                                                        gammaEnergy);
  vdp->push_back(g);
  
  G4double totMomentum = sqrt(kineticEnergy*(totalEnergy + electron_mass_c2));
  G4ThreeVector dir = totMomentum*direction - gammaEnergy*gammaDirection;
  direction = dir.unit();

  // energy of primary
  G4double finalE = kineticEnergy - gammaEnergy;

  // stop tracking and create new secondary instead of primary
  if(gammaEnergy > SecondaryThreshold()) {
    fParticleChange->ProposeTrackStatus(fStopAndKill);
    fParticleChange->SetProposedKineticEnergy(0.0);
    G4DynamicParticle* el = 
      new G4DynamicParticle(const_cast<G4ParticleDefinition*>(particle),
			    direction, finalE);
    vdp->push_back(el);

    // continue tracking
  } else {
    fParticleChange->SetProposedMomentumDirection(direction);
    fParticleChange->SetProposedKineticEnergy(finalE);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......



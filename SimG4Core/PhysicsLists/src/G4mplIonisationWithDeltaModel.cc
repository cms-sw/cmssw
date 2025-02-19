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
// $Id: G4mplIonisationWithDeltaModel.cc,v 1.1 2010/07/29 23:05:19 sunanda Exp $
// GEANT4 tag $Name: V01-07-04-01 $
//
// -------------------------------------------------------------------
//
// GEANT4 Class header file
//
//
// File name:     G4mplIonisationWithDeltaModel
//
// Author:        Vladimir Ivanchenko 
//
// Creation date: 06.09.2005
//
// Modifications:
// 12.08.2007 Changing low energy approximation and extrapolation. 
//            Small bug fixing and refactoring (M. Vladymyrov)
// 13.11.2007 Use low-energy asymptotic from [3] (V.Ivanchenko) 
//
//
// -------------------------------------------------------------------
// References
// [1] Steven P. Ahlen: Energy loss of relativistic heavy ionizing particles, 
//     S.P. Ahlen, Rev. Mod. Phys 52(1980), p121
// [2] K.A. Milton arXiv:hep-ex/0602040
// [3] S.P. Ahlen and K. Kinoshita, Phys. Rev. D26 (1982) 2347


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "G4mplIonisationWithDeltaModel.hh"
#include "Randomize.hh"
#include "G4LossTableManager.hh"
#include "G4ParticleChangeForLoss.hh"
#include "G4Electron.hh"
#include "G4DynamicParticle.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

using namespace std;

G4mplIonisationWithDeltaModel::G4mplIonisationWithDeltaModel(G4double mCharge, const G4String& nam)
  : G4VEmModel(nam),G4VEmFluctuationModel(nam),
  magCharge(mCharge),
  twoln10(log(100.0)),
  betalow(0.01),
  betalim(0.1),
  beta2lim(betalim*betalim),
  bg2lim(beta2lim*(1.0 + beta2lim))
{
  nmpl = G4int(abs(magCharge) * 2 * fine_structure_const + 0.5);
  if(nmpl > 6)      { nmpl = 6; }
  else if(nmpl < 1) { nmpl = 1; }
  pi_hbarc2_over_mc2 = pi * hbarc * hbarc / electron_mass_c2;
  chargeSquare = magCharge * magCharge;
  dedxlim = 45.*nmpl*nmpl*GeV*cm2/g;
  fParticleChange = 0;
  theElectron = G4Electron::Electron();
  G4cout << "### Monopole ionisation model with d-electron production, Gmag= " 
	 << magCharge/eplus << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4mplIonisationWithDeltaModel::~G4mplIonisationWithDeltaModel()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void 
G4mplIonisationWithDeltaModel::Initialise(const G4ParticleDefinition* p,
					  const G4DataVector&)
{
  monopole = p;
  mass     = monopole->GetPDGMass();
  if(!fParticleChange) { fParticleChange = GetParticleChangeForLoss(); }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double 
G4mplIonisationWithDeltaModel::ComputeDEDXPerVolume(const G4Material* material,
						    const G4ParticleDefinition* p,
						    G4double kineticEnergy,
						    G4double maxEnergy)
{
  G4double tmax = MaxSecondaryEnergy(p,kineticEnergy);
  G4double cutEnergy = std::min(tmax, maxEnergy);
  G4double tau   = kineticEnergy / mass;
  G4double gam   = tau + 1.0;
  G4double bg2   = tau * (tau + 2.0);
  G4double beta2 = bg2 / (gam * gam);
  G4double beta  = sqrt(beta2);

  // low-energy asymptotic formula
  G4double dedx  = dedxlim*beta*material->GetDensity();

  // above asymptotic
  if(beta > betalow) {

    // high energy
    if(beta >= betalim) {
      dedx = ComputeDEDXAhlen(material, bg2, cutEnergy);

    } else {

      G4double dedx1 = dedxlim*betalow*material->GetDensity();
      G4double dedx2 = ComputeDEDXAhlen(material, bg2lim, cutEnergy);

      // extrapolation between two formula 
      G4double kapa2 = beta - betalow;
      G4double kapa1 = betalim - beta;
      dedx = (kapa1*dedx1 + kapa2*dedx2)/(kapa1 + kapa2);
    }
  }
  return dedx;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double 
G4mplIonisationWithDeltaModel::ComputeDEDXAhlen(const G4Material* material, 
						G4double bg2, 
						G4double cutEnergy)
{
  G4double eDensity = material->GetElectronDensity();
  G4double eexc  = material->GetIonisation()->GetMeanExcitationEnergy();

  // Ahlen's formula for nonconductors, [1]p157, f(5.7)
  G4double dedx = 
    0.5*(log(2.0 * electron_mass_c2 * bg2*cutEnergy / (eexc*eexc)) - 1.0);

  // Kazama et al. cross-section correction
  G4double  k = 0.406;
  if(nmpl > 1) { k = 0.346; }

  // Bloch correction
  const G4double B[7] = { 0.0, 0.248, 0.672, 1.022, 1.243, 1.464, 1.685}; 

  dedx += 0.5 * k - B[nmpl];

  // density effect correction
  G4double x = log(bg2)/twoln10;
  dedx -= material->GetIonisation()->DensityCorrection(x);

  // now compute the total ionization loss
  dedx *=  pi_hbarc2_over_mc2 * eDensity * nmpl * nmpl;

  if (dedx < 0.0) { dedx = 0; }
  return dedx;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double
G4mplIonisationWithDeltaModel::ComputeCrossSectionPerElectron(
                                           const G4ParticleDefinition* p,
					   G4double kineticEnergy,
					   G4double cutEnergy,
					   G4double maxKinEnergy)
{
  G4double cross = 0.0;
  G4double tmax = MaxSecondaryEnergy(p, kineticEnergy);
  G4double maxEnergy = min(tmax,maxKinEnergy);
  if(cutEnergy < maxEnergy) {
    cross = (1.0/cutEnergy - 1.0/maxEnergy)*twopi_mc2_rcl2*chargeSquare;
  }
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double 
G4mplIonisationWithDeltaModel::ComputeCrossSectionPerAtom(
					  const G4ParticleDefinition* p,
					  G4double kineticEnergy,
					  G4double Z, G4double,
					  G4double cutEnergy,
					  G4double maxEnergy)
{
  G4double cross = 
    Z*ComputeCrossSectionPerElectron(p,kineticEnergy,cutEnergy,maxEnergy);
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void 
G4mplIonisationWithDeltaModel::SampleSecondaries(vector<G4DynamicParticle*>* vdp,
						 const G4MaterialCutsCouple*,
						 const G4DynamicParticle* dp,
						 G4double minKinEnergy,
						 G4double maxEnergy)
{
  G4double kineticEnergy = dp->GetKineticEnergy();
  G4double tmax = MaxSecondaryEnergy(dp->GetDefinition(),kineticEnergy);

  G4double maxKinEnergy = std::min(maxEnergy,tmax);
  if(minKinEnergy >= maxKinEnergy) { return; }

  //G4cout << "G4mplIonisationWithDeltaModel::SampleSecondaries: E(GeV)= "
  //	 << kineticEnergy/GeV << " M(GeV)= " << mass/GeV
  //	 << " tmin(MeV)= " << minKinEnergy/MeV << G4endl;

  G4double totEnergy     = kineticEnergy + mass;
  G4double etot2         = totEnergy*totEnergy;
  G4double beta2         = kineticEnergy*(kineticEnergy + 2.0*mass)/etot2;
  
  // sampling without nuclear size effect
  G4double q = G4UniformRand();
  G4double deltaKinEnergy = minKinEnergy*maxKinEnergy
    /(minKinEnergy*(1.0 - q) + maxKinEnergy*q);

  // delta-electron is produced
  G4double totMomentum = totEnergy*sqrt(beta2);
  G4double deltaMomentum =
           sqrt(deltaKinEnergy * (deltaKinEnergy + 2.0*electron_mass_c2));
  G4double cost = deltaKinEnergy * (totEnergy + electron_mass_c2) /
                                   (deltaMomentum * totMomentum);
  if(cost > 1.0) { cost = 1.0; }

  G4double sint = sqrt((1.0 - cost)*(1.0 + cost));

  G4double phi = twopi * G4UniformRand() ;

  G4ThreeVector deltaDirection(sint*cos(phi),sint*sin(phi), cost);
  G4ThreeVector direction = dp->GetMomentumDirection();
  deltaDirection.rotateUz(direction);

  // create G4DynamicParticle object for delta ray
  G4DynamicParticle* delta = 
    new G4DynamicParticle(theElectron,deltaDirection,deltaKinEnergy);

  vdp->push_back(delta);

  // Change kinematics of primary particle
  kineticEnergy       -= deltaKinEnergy;
  G4ThreeVector finalP = direction*totMomentum - deltaDirection*deltaMomentum;
  finalP               = finalP.unit();

  fParticleChange->SetProposedKineticEnergy(kineticEnergy);
  fParticleChange->SetProposedMomentumDirection(finalP);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4mplIonisationWithDeltaModel::SampleFluctuations(
				       const G4Material* material,
				       const G4DynamicParticle* dp,
				       G4double& tmax,
				       G4double& length,
				       G4double& meanLoss)
{
  G4double siga = Dispersion(material,dp,tmax,length);
  G4double loss = meanLoss;
  siga = sqrt(siga);
  G4double twomeanLoss = meanLoss + meanLoss;

  if(twomeanLoss < siga) {
    G4double x;
    do {
      loss = twomeanLoss*G4UniformRand();
      x = (loss - meanLoss)/siga;
    } while (1.0 - 0.5*x*x < G4UniformRand());
  } else {
    do {
      loss = G4RandGauss::shoot(meanLoss,siga);
    } while (0.0 > loss || loss > twomeanLoss);
  }
  return loss;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double 
G4mplIonisationWithDeltaModel::Dispersion(const G4Material* material,
					  const G4DynamicParticle* dp,
					  G4double& tmax,
					  G4double& length)
{
  G4double siga = 0.0;
  G4double tau   = dp->GetKineticEnergy()/mass;
  if(tau > 0.0) { 
    G4double electronDensity = material->GetElectronDensity();
    G4double gam   = tau + 1.0;
    G4double invbeta2 = (gam*gam)/(tau * (tau+2.0));
    siga  = (invbeta2 - 0.5) * twopi_mc2_rcl2 * tmax * length
      * electronDensity * chargeSquare;
  }
  return siga;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double 
G4mplIonisationWithDeltaModel::MaxSecondaryEnergy(const G4ParticleDefinition*,
						  G4double kinEnergy)
{
  G4double tau = kinEnergy/mass;
  return 2.0*electron_mass_c2*tau*(tau + 2.);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

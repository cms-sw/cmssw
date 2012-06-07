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
// File name:     G4SeltzerBergerModel95
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

#include "G4SeltzerBergerModel95.hh"
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

#include "G4Physics2DVector95.hh"

#include "G4ios.hh"
#include <fstream>
#include <iomanip>


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

using namespace std;

G4SeltzerBergerModel95::G4SeltzerBergerModel95(const G4ParticleDefinition* p,
					   const G4String& name)
  : G4eBremsstrahlungRelModel95(p,name)
{
  SetLowEnergyLimit(0.0);
  SetLPMFlag(false);
  dataSB.resize(101,0);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4SeltzerBergerModel95::~G4SeltzerBergerModel95()
{
  for(size_t i=0; i<101; ++i) { delete dataSB[i]; }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4SeltzerBergerModel95::Initialise(const G4ParticleDefinition* p,
				      const G4DataVector& cuts)
{
  // check environment variable
  // Build the complete string identifying the file with the data set
  char* path = getenv("G4LEDATA");

  // Access to elements
  const G4ElementTable* theElmTable = G4Element::GetElementTable();
  size_t numOfElm = G4Element::GetNumberOfElements();
  if(numOfElm > 0) {
    for(size_t i=0; i<numOfElm; ++i) {
      G4int Z = G4int(((*theElmTable)[i])->GetZ());
      if(Z < 1)        { Z = 1; }
      else if(Z > 100) { Z = 100; }
      //G4cout << "Z= " << Z << G4endl;
      // Initialisation
      if(!dataSB[Z]) { ReadData(Z, path); }
    }
  }

  G4eBremsstrahlungRelModel95::Initialise(p, cuts);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4SeltzerBergerModel95::ReadData(size_t Z, const char* path)
{
  //  G4cout << "ReadData Z= " << Z << G4endl;
  // G4cout << "Status for Z= " << dataSB[Z] << G4endl;
  //if(path) { G4cout << path << G4endl; }
  if(dataSB[Z]) { return; }
  const char* datadir = path;

  if(!datadir) {
    datadir = getenv("G4LEDATA");
    if(!datadir) {
      //G4Exception("G4SeltzerBergerModel95::ReadData() Environment variable G4LEDATA not defined");
      //G4Exception("G4SeltzerBergerModel95::ReadData()","em0006",FatalException,
      //		  "Environment variable G4LEDATA not defined");
      return;
    }
  }
  std::ostringstream ost;
  ost << datadir << "/brem_SB/br" << Z;
  std::ifstream fin(ost.str().c_str());
  if( !fin.is_open()) {
    //G4ExceptionDescription ed;
    G4cout << "Bremsstrahlung data file <" << ost.str().c_str()
	   << "> is not opened!" << G4endl;
    //G4Exception("G4SeltzerBergerModel95::ReadData() G4LEDATA version should be G4EMLOW6.23 or later.");
    return;
  } 
  //G4cout << "G4SeltzerBergerModel95 read from <" << ost.str().c_str() 
  //	 << ">" << G4endl;
  G4Physics2DVector95* v = new G4Physics2DVector95();
  if(v->Retrieve(fin)) { dataSB[Z] = v; }
  else {
    //G4ExceptionDescription ed;
    G4cout << "Bremsstrahlung data file <" << ost.str().c_str()
       << "> is not retrieved!" << G4endl;
    //G4Exception("G4SeltzerBergerModel95::ReadData() G4LEDATA version should be G4EMLOW6.23 or later.");
    delete v;
  }
  // G4cout << dataSB[Z] << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4SeltzerBergerModel95::ComputeDXSectionPerAtom(G4double gammaEnergy)
{

  if(gammaEnergy < 0.0 || kinEnergy <= 0.0) { return 0.0; }
  G4double x = gammaEnergy/kinEnergy;
  G4double y = log(kinEnergy/MeV);
  G4int Z = G4int(currentZ);
  //G4cout << "G4SeltzerBergerModel95::ComputeDXSectionPerAtom Z= " << Z
  //	 << " x= " << x << " y= " << y << " " << dataSB[Z] << G4endl;
  if(!dataSB[Z]) { ReadData(Z); }
  G4double invb2 = totalEnergy*totalEnergy/(kinEnergy*(kinEnergy + 2*particleMass));
  G4double cross = dataSB[Z]->Value(x,y)*invb2*millibarn/bremFactor;
  
  if(!isElectron) {
    if(1 - x < 1.e-20) { cross = 0.0; }
    else {
      G4double invbeta1 = sqrt(invb2);
      G4double e2 = kinEnergy - gammaEnergy;
      G4double invbeta2 = (e2 + particleMass)/sqrt(e2*(e2 + 2*particleMass)); 
      cross *= exp(twopi*fine_structure_const*currentZ*(invbeta1 - invbeta2)); 
    }
  }
  
  return cross;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void 
G4SeltzerBergerModel95::SampleSecondaries(std::vector<G4DynamicParticle*>* vdp, 
					const G4MaterialCutsCouple* couple,
					const G4DynamicParticle* dp,
					G4double cutEnergy,
					G4double maxEnergy)
{
  G4double kineticEnergy = dp->GetKineticEnergy();
  G4double cut  = std::min(cutEnergy, kineticEnergy);
  G4double emax = std::min(maxEnergy, kineticEnergy);
  if(cut >= emax) { return; }

  SetupForMaterial(particle, couple->GetMaterial(), kineticEnergy);

  const G4Element* elm = 
    SelectRandomAtom(couple,particle,kineticEnergy,cut,emax);
  SetCurrentElement(elm->GetZ());
  G4int Z = G4int(currentZ);

  totalEnergy = kineticEnergy + particleMass;
  densityCorr = densityFactor*totalEnergy*totalEnergy;
  G4double totMomentum = sqrt(kineticEnergy*(totalEnergy + electron_mass_c2));
  G4ThreeVector direction = dp->GetMomentumDirection();
  /*
  G4cout << "G4SeltzerBergerModel95::SampleSecondaries E(MeV)= " 
	 << kineticEnergy/MeV
	 << " Z= " << Z << " cut(MeV)= " << cut/MeV 
	 << " emax(MeV)= " << emax/MeV << " corr= " << densityCorr << G4endl;
  */
  G4double xmin = log(cut*cut + densityCorr);
  G4double xmax = log(emax*emax  + densityCorr);
  G4double y = log(kineticEnergy/MeV);

  G4double gammaEnergy, v; 

  // majoranta
  G4double vmax = dataSB[Z]->Value(cut/kineticEnergy, y);
  if(isElectron && Z > 12 && kineticEnergy < 100*keV) {
    if((Z < 41 && kineticEnergy < 10*keV) ||
       (Z >= 41 && Z < 61 && kineticEnergy < 50*keV) ||
       (Z >= 61) )
      {
	v = 1.05*dataSB[Z]->Value(emax/kineticEnergy, y);
	if(v > vmax) { vmax = v; }
      }
  }

  //G4cout<<"y= "<<y<<" xmin= "<<xmin<<" xmax= "<<xmax<<" vmax= "<<vmax<<G4endl;

  do {
    G4double x = exp(xmin + G4UniformRand()*(xmax - xmin)) - densityCorr;
    if(x < 0.0) { x = 0.0; }
    gammaEnergy = sqrt(x);
    G4double x1 = gammaEnergy/kineticEnergy;
    v = dataSB[Z]->Value(x1, y);
        
    if(!isElectron) {
      if(1 - x1 < 1.e-20) { v = 0.0; }
      else {
	G4double e1 = kineticEnergy - cut;
	G4double invbeta1 = (e1 + particleMass)/sqrt(e1*(e1 + 2*particleMass));
	G4double e2 = kineticEnergy - gammaEnergy;
	G4double invbeta2 = (e2 + particleMass)/sqrt(e2*(e2 + 2*particleMass));
	v *= exp(twopi*fine_structure_const*currentZ*(invbeta1 - invbeta2));
      } 
    }
   
    if ( v > 1.5*vmax ) {
      G4cout << "### G4SeltzerBergerModel95 Warning: Majoranta exceeded! "
	     << v << " > " << vmax
	     << " Egamma(MeV)= " << gammaEnergy
	     << " Ee(MeV)= " << kineticEnergy
	     << " Z= " << Z << "  " << particle->GetParticleName()
	     << G4endl;
    }

  } while (v < vmax*G4UniformRand());

  //
  // angles of the emitted gamma. ( Z - axis along the parent particle)
  // use general interface
  //
  G4double theta = 
    GetAngularDistribution()->PolarAngle(totalEnergy,totalEnergy-gammaEnergy,Z);

  G4double sint = sin(theta);
  G4double phi = twopi * G4UniformRand();
  G4ThreeVector gammaDirection(sint*cos(phi),sint*sin(phi), cos(theta));
  gammaDirection.rotateUz(direction);

  // create G4DynamicParticle object for the Gamma
  G4DynamicParticle* g = 
    new G4DynamicParticle(theGamma,gammaDirection,gammaEnergy);
  vdp->push_back(g);
  
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



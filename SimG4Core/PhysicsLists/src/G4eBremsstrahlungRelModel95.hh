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
// GEANT4 Class header file
//
//
// File name:     G4eBremsstrahlungRelModel95
//                extention of standard G4eBremsstrahlungModel
//
// Author:        Andreas Schaelicke 
//
// Creation date: 28.03.2008
//
// Modifications:
//
//
// Class Description:
//
// Implementation of energy loss for gamma emission by electrons and
// positrons including an improved version of the LPM effect

// -------------------------------------------------------------------
//

#ifndef G4eBremsstrahlungRelModel95_h
#define G4eBremsstrahlungRelModel95_h 1

#include "G4VEmModel.hh"
#include "G4NistManager.hh"

class G4ParticleChangeForLoss;
class G4PhysicsVector;

class G4eBremsstrahlungRelModel95 : public G4VEmModel
{

public:

  G4eBremsstrahlungRelModel95(const G4ParticleDefinition* p = 0, 
			    const G4String& nam = "eBremLPM");

  virtual ~G4eBremsstrahlungRelModel95();

  virtual void Initialise(const G4ParticleDefinition*, const G4DataVector&);

  virtual G4double ComputeDEDXPerVolume(const G4Material*,
					const G4ParticleDefinition*,
					G4double kineticEnergy,
					G4double cutEnergy);
					
  virtual G4double ComputeCrossSectionPerAtom(const G4ParticleDefinition*,
					      G4double tkin, 
					      G4double Z,   G4double,
					      G4double cutEnergy,
					      G4double maxEnergy = DBL_MAX);
  
  virtual void SampleSecondaries(std::vector<G4DynamicParticle*>*,
				 const G4MaterialCutsCouple*,
				 const G4DynamicParticle*,
				 G4double cutEnergy,
				 G4double maxEnergy);

  virtual void SetupForMaterial(const G4ParticleDefinition*,
                                const G4Material*,G4double);

  inline void SetLPMconstant(G4double val);
  inline G4double LPMconstant() const;

protected:

  virtual G4double ComputeDXSectionPerAtom(G4double gammaEnergy);

  // * fast inline functions *
  inline void SetCurrentElement(const G4double);

private:

  void InitialiseConstants();

  void CalcLPMFunctions(G4double gammaEnergy);

  G4double ComputeBremLoss(G4double cutEnergy);

  G4double ComputeXSectionPerAtom(G4double cutEnergy);

  G4double ComputeRelDXSectionPerAtom(G4double gammaEnergy);

  void SetParticle(const G4ParticleDefinition* p);

  inline G4double Phi1(G4double,G4double);
  inline G4double Phi1M2(G4double,G4double);
  inline G4double Psi1(G4double,G4double);
  inline G4double Psi1M2(G4double,G4double);

  // hide assignment operator
  G4eBremsstrahlungRelModel95 & operator=(const  G4eBremsstrahlungRelModel95 &right);
  G4eBremsstrahlungRelModel95(const  G4eBremsstrahlungRelModel95&);

protected:

  G4NistManager*              nist;
  const G4ParticleDefinition* particle;
  G4ParticleDefinition*       theGamma;
  G4ParticleChangeForLoss*    fParticleChange;

  G4double bremFactor;

  // cash
  G4double particleMass;
  G4double kinEnergy;
  G4double totalEnergy;
  G4double currentZ;
  //G4double z13, z23, lnZ;
  //G4double Fel, Finel, fCoulomb, fMax; 
  G4double densityFactor;
  G4double densityCorr;

  G4bool   isElectron;

private:

  static const G4double xgi[8], wgi[8];
  static const G4double Fel_light[5];
  static const G4double Finel_light[5];

  // consts
  G4double lowKinEnergy;
  G4double fMigdalConstant;
  G4double fLPMconstant;
  G4double energyThresholdLPM;
  G4double facFel, facFinel;
  G4double preS1,logTwo;

  // cash
  //G4double particleMass;
  //G4double kinEnergy;
  //G4double totalEnergy;
  //G4double currentZ;
  G4double z13, z23, lnZ;
  G4double Fel, Finel, fCoulomb, fMax; 
  //G4double densityFactor;
  //G4double densityCorr;

  // LPM effect
  G4double lpmEnergy;
  G4PhysicsVector  *fXiLPM, *fPhiLPM, *fGLPM;
  G4double xiLPM, phiLPM, gLPM;

  // critical gamma energies
  G4double klpm, kp;

  // flags
  G4bool   use_completescreening;
  G4bool   isInitialised;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline void G4eBremsstrahlungRelModel95::SetCurrentElement(const G4double Z)
{
  if(Z != currentZ) {
    currentZ = Z;

    G4int iz = G4int(Z);
    z13 = nist->GetZ13(iz);
    z23 = z13*z13;
    lnZ = nist->GetLOGZ(iz);

    if (iz <= 4) {
      Fel = Fel_light[iz];  
      Finel = Finel_light[iz] ; 
    }
    else {
      Fel = facFel - lnZ/3. ;
      Finel = facFinel - 2.*lnZ/3. ;
    }

    fCoulomb = GetCurrentElement()->GetfCoulomb();
    fMax = Fel-fCoulomb + Finel/currentZ  +  (1.+1./currentZ)/12.;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


inline G4double G4eBremsstrahlungRelModel95::Phi1(G4double gg, G4double)
{
  //       Thomas-Fermi FF from Tsai, eq.(3.38) for Z>=5
  return 20.863 - 2.*std::log(1. + sqr(0.55846*gg) )
    - 4.*( 1. - 0.6*std::exp(-0.9*gg) - 0.4*std::exp(-1.5*gg) );
}

inline G4double G4eBremsstrahlungRelModel95::Phi1M2(G4double gg, G4double)
{
  //       Thomas-Fermi FF from Tsai, eq. (3.39) for Z>=5
  // return Phi1(gg,Z) - 
  return 2./(3.*(1. + 6.5*gg +6.*gg*gg) );
}

inline G4double G4eBremsstrahlungRelModel95::Psi1(G4double eps, G4double)
{
  //       Thomas-Fermi FF from Tsai, eq.(3.40) for Z>=5 
  return 28.340 - 2.*std::log(1. + sqr(3.621*eps) )
    - 4.*( 1. - 0.7*std::exp(-8*eps) - 0.3*std::exp(-29.*eps) );
}

inline G4double G4eBremsstrahlungRelModel95::Psi1M2(G4double eps, G4double)
{
  //       Thomas-Fermi FF from Tsai, eq. (3.41) for Z>=5
  return  2./(3.*(1. + 40.*eps +400.*eps*eps) );
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline 
void G4eBremsstrahlungRelModel95::SetLPMconstant(G4double val) 
{
  fLPMconstant = val;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline 
G4double G4eBremsstrahlungRelModel95::LPMconstant() const 
{
  return fLPMconstant;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


#endif

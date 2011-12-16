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
// $Id: G4SeltzerBergerModel95.hh,v 1.14 2010-10-26 10:35:22 vnivanch Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// -------------------------------------------------------------------
//
// GEANT4 Class header file
//
//
// File name:     G4SeltzerBergerModel95
//
// Author:        Andreas Schaelicke & Vladimir Ivantchenko
//
// Creation date: 25.09.2011
//
// Modifications:
//
//
// Class Description:
//
// Implementation of the bremssrahlung model using
// S.M. Seltzer and M.J. Berger Atomic data and Nuclear Data 
// Tables 35 (1986) 345

// -------------------------------------------------------------------
//

#ifndef G4SeltzerBergerModel95_h
#define G4SeltzerBergerModel95_h 1

#include "G4eBremsstrahlungRelModel95.hh"
#include "globals.hh"

class G4Physics2DVector95;

class G4SeltzerBergerModel95 : public G4eBremsstrahlungRelModel95
{

public:

  G4SeltzerBergerModel95(const G4ParticleDefinition* p = 0, 
		       const G4String& nam = "eBremSB");

  virtual ~G4SeltzerBergerModel95();

  virtual void Initialise(const G4ParticleDefinition*, const G4DataVector&);

  virtual void SampleSecondaries(std::vector<G4DynamicParticle*>*,
				 const G4MaterialCutsCouple*,
				 const G4DynamicParticle*,
				 G4double cutEnergy,
				 G4double maxEnergy);

protected:

  virtual G4double ComputeDXSectionPerAtom(G4double gammaEnergy);

private:

  void ReadData(size_t Z, const char* path = 0);

  // hide assignment operator
  G4SeltzerBergerModel95 & operator=(const  G4SeltzerBergerModel95 &right);
  G4SeltzerBergerModel95(const  G4SeltzerBergerModel95&);

  std::vector<G4Physics2DVector95*> dataSB;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


#endif

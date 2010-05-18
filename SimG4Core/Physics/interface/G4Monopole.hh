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
#ifndef G4Monopole_h
#define G4Monopole_h 1

//---------------------------------------------------------------------------
//
//---------------------------------------------------------------------------
//
// ClassName:   G4Monopole
//
// Authors:   21.03.05  V.Ivanchenko 
//
// Modified:
//
//----------------------------------------------------------------------------
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "G4ParticleDefinition.hh"
#include "globals.hh"

// ######################################################################
// ###                       Monopole                                 ###
// ######################################################################

class G4Monopole : public G4ParticleDefinition
{
private:

  static G4Monopole*  theMonopole;

  G4Monopole(
       const G4String&     aName,        G4double            mass,
       G4double            width,        G4double            charge,   
       G4int               iSpin,        G4int               iParity,    
       G4int               iConjugation, G4int               iIsospin,   
       G4int               iIsospin3,    G4int               gParity,
       const G4String&     pType,        G4int               lepton,      
       G4int               baryon,       G4int               encoding,
       G4bool              stable,       G4double            lifetime,
       G4DecayTable        *decaytable );

  virtual ~G4Monopole();

public: 
  
  static G4Monopole* MonopoleDefinition(G4double mass_ = 100.*GeV, 
                                        G4int magCharge_ = 1, 
                                        G4int elCharge_  = 0,
					G4int pdgEncoding= 0);

  static G4Monopole* Monopole();

  static G4double MagneticCharge() {return magCharge;};

private:

  static G4double magCharge;
};

#endif

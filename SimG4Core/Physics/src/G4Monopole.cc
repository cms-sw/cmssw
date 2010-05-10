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
//---------------------------------------------------------------------------
//
// ClassName:   G4Monopole
//  
// Description: 
//
// Authors:   21.03.05  V.Ivanchenko 
//
// Modified:
//
//----------------------------------------------------------------------------
//

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#include "SimG4Core/Physics/interface/G4Monopole.hh"
#include "G4ParticleTable.hh"

// ######################################################################
// ###                        Monopole                                ###
// ######################################################################

G4Monopole* G4Monopole::theMonopole = 0;
G4double    G4Monopole::magCharge = 0.0;

G4Monopole::G4Monopole(
       const G4String&     aName,        G4double            mass,
       G4double            width,        G4double            charge,   
       G4int               iSpin,        G4int               iParity,    
       G4int               iConjugation, G4int               iIsospin,   
       G4int               iIsospin3,    G4int               gParity,
       const G4String&     pType,        G4int               lepton,      
       G4int               baryon,       G4int               encoding,
       G4bool              stable,       G4double            lifetime,
       G4DecayTable        *decaytable)
 : G4ParticleDefinition( aName, mass, width, charge, iSpin, iParity,
           iConjugation, iIsospin, iIsospin3, gParity, pType,
           lepton, baryon, encoding, stable, lifetime, decaytable )
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4Monopole::~G4Monopole()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
//     
//    Arguments for constructor are as follows
//               name             mass          width         charge
//             2*spin           parity  C-conjugation
//          2*Isospin       2*Isospin3       G-parity
//               type    lepton number  baryon number   PDG encoding
//             stable         lifetime    decay table 
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4Monopole* G4Monopole::MonopoleDefinition(G4double mass, G4int mCharge, G4int eCharge, G4int pdgEncoding)
{    
  if(!theMonopole) {
    magCharge = eplus * G4double(mCharge) / fine_structure_const * 0.5;
    theMonopole = new G4Monopole(
              "monopole",         mass,       0.0*MeV,       eplus*G4double(eCharge), 
		    0,               0,             0,          
		    0,               0,             0,             
	      "boson",               0,             0,       pdgEncoding,
	         true,            -1.0,             0);


    G4cout << "Monopole is created: m(GeV)= " << theMonopole->GetPDGMass()/GeV 
	   << " Qel= " << theMonopole->GetPDGCharge()/eplus
	   << " Qmag= " << magCharge/eplus
	   << " PDG encoding = " << pdgEncoding
	   << G4endl;
  }
  return theMonopole;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4Monopole* G4Monopole::Monopole()
{    
  if(!theMonopole) theMonopole = MonopoleDefinition();
  return theMonopole;
} 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....


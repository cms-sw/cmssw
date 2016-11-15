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
// $Id: CMSDarkPhotonModel.cc 74581 2013-10-15 12:03:25Z gcosmo $
//
// -------------------------------------------------------------------
//
// GEANT4 Class file
//
//
// File name:     CMSDarkPairProduction
//
// Author:        Dustin Stolp (dostolp@ucdavis.edu)
//                Sushil S. Chauhan (schauhan@cern.ch)  
// Creation date: 01.22.2015
//
// Modifications:
//
// Class Description:
//
// Main References:
//  J.W.Motz et.al., Rev. Mod. Phys. 41 (1969) 581.
//  S.Klein,  Rev. Mod. Phys. 71 (1999) 1501.
//  T.Stanev et.al., Phys. Rev. D25 (1982) 1291.
//  M.L.Ter-Mikaelian, High-energy Electromagnetic Processes in Condensed Media,
//                     Wiley, 1972.
//
// -------------------------------------------------------------------
//
#ifndef CMSDarkPairProduction_h
#define CMSDarkPairProduction_h 1

#include <CLHEP/Units/PhysicalConstants.h>

#include "G4PairProductionRelModel.hh"
#include "G4PhysicsTable.hh"
#include "G4NistManager.hh"
#include "G4VEmModel.hh"

class CMSDarkPairProduction : public G4PairProductionRelModel
{
  //G4double dark_factor; 
public:
  CMSDarkPairProduction(const G4ParticleDefinition* p = 0,
		      const G4double df = 1E0,
                      const G4String& nam = "BetheHeitlerLPM");

  virtual ~CMSDarkPairProduction();

  virtual G4double ComputeCrossSectionPerAtom(
                      const G4ParticleDefinition*,
                      G4double kinEnergy,
                      G4double Z,
                      G4double A=0.,
                      G4double cut=0.,
                      G4double emax=DBL_MAX);

  void SampleSecondaries(std::vector<G4DynamicParticle*>* fvect,
               	      const G4MaterialCutsCouple* couple,
               	      const G4DynamicParticle* aDynamicGamma,
                      G4double e1,
                      G4double e2); 

private:
  G4int count=0;
};
#endif

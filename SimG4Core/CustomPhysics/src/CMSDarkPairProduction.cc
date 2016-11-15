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
// File name:     CMSDarkPhotonModel
//
// Authors of this file:        Dustin Stolp (dostolp@ucdavis.edu)
//                              Sushil S. Chauhan (schauhan@cern.ch) 
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
#include "SimG4Core/CustomPhysics/interface/CMSDarkPairProduction.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4PairProductionRelModel.hh"

using namespace std;

static const G4double xsfactor =
  4*fine_structure_const*classic_electr_radius*classic_electr_radius;

static G4double dark_factor;

CMSDarkPairProduction::CMSDarkPairProduction(const G4ParticleDefinition* p,G4double df,const G4String& nam) : G4PairProductionRelModel(p,nam){
  dark_factor = df;

}

CMSDarkPairProduction::~CMSDarkPairProduction(){}

G4double CMSDarkPairProduction::ComputeCrossSectionPerAtom(const G4ParticleDefinition*,
                                                     G4double gammaEnergy, G4double Z,
                                                     G4double, G4double, G4double)
{
  count++;
//  gammaEnergy = gammaEnergy*1E5;
  
  G4double crossSection = 0.0 ;
  //  if ( Z < 0.9 ) return crossSection;
  if ( gammaEnergy <= 2.0*electron_mass_c2 ) return crossSection;
  
  SetCurrentElement(Z);
  // choose calculator according to parameters and switches
  // in the moment only one calculator:
  crossSection=ComputeXSectionPerAtom(gammaEnergy,Z);
  
  G4double xi = Finel/(Fel - fCoulomb); // inelastic contribution
  crossSection *= dark_factor * xsfactor*Z*(Z+xi);
  return crossSection;
}
void
CMSDarkPairProduction::SampleSecondaries(std::vector<G4DynamicParticle*>* fvect,
               const G4MaterialCutsCouple* couple,
               const G4DynamicParticle* aDynamicGamma,
               G4double e1,
               G4double e2)
{
G4PairProductionRelModel::SampleSecondaries(fvect, couple, aDynamicGamma, e1, e2);
        
}


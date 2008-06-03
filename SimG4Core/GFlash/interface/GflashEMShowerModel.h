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
//
// $Id: GflashEMShowerModel.h,v 1.4 2008/02/29 23:40:55 syjun Exp $
// GEANT4 tag $Name:  $
//
//
//---------------------------------------------------------------
//  GEANT 4 class header file
//
//  GflashEMShowerModel
//
//  Class description:
//
//  GFlash parameterisation shower model.

// Authors: E.Barberio & Joanna Weng - 9.11.04
// other authors : Soon Yung Jun & Dongwook Jang - 2007/12/07
//---------------------------------------------------------------
#ifndef GflashEMShowerModel_h
#define GflashEMShowerModel_h

#include "G4VFastSimulationModel.hh"
#include "G4TouchableHandle.hh"
#include "G4Navigator.hh"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GflashEMShowerProfile;
class G4Step;

class GflashEMShowerModel : public G4VFastSimulationModel {

 public:
  
  GflashEMShowerModel (G4String name, G4Envelope* env, edm::ParameterSet parSet);
  ~GflashEMShowerModel ();  

  G4bool ModelTrigger(const G4FastTrack &); 
  G4bool IsApplicable(const G4ParticleDefinition&);
  void DoIt(const G4FastTrack&, G4FastStep&);

  G4bool excludeDetectorRegion(const G4FastTrack& fastTrack);

private:

  edm::ParameterSet theParSet;
  GflashEMShowerProfile *theProfile;
  G4Step *theGflashStep;
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

};
#endif

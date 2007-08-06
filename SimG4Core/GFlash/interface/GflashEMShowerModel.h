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
// $Id: GflashEMShowerModel.h,v 1.1 2007/05/15 23:16:40 syjun Exp $
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
//---------------------------------------------------------------
#ifndef GflashEMShowerModel_h
#define GflashEMShowerModel_h 1

//G4 Standard
#include "G4VFastSimulationModel.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ios.hh"

//GFlash
#include "SimG4Core/GFlash/interface/GflashEMShowerModelMessenger.h"
#include "SimG4Core/G4gflash/src/GFlashHitMaker.hh"

#include "GFlashParticleBounds.hh"
#include "GFlashEnergySpot.hh"
//#include "GFlashHitMaker.hh"
#include  <vector>

class GVFlashShowerParameterisation;
class GFlashHomoShowerParameterisation;
class GFlashSamplingShowerParameterisation;

class GflashEMShowerModel : public G4VFastSimulationModel
{
  public:  // with description

    GflashEMShowerModel (G4String, G4Envelope*);
    GflashEMShowerModel (G4String);
    ~GflashEMShowerModel ();  
      // Constructors, destructor
  int model_trigger;
  int isapp;
  int edoit;	 
  G4ThreeVector test;	

    G4bool ModelTrigger(const G4FastTrack &); 
    G4bool IsApplicable(const G4ParticleDefinition&);
    void DoIt(const G4FastTrack&, G4FastStep&);
      // Checks whether conditions of fast parameterisation are fullfilled
  
    // setting

    inline void SetFlagParamType(G4int I)
      { FlagParamType = I; }
    inline void SetFlagParticleContainment(G4int I)
      { FlagParticleContainment = I; }
    inline void SetStepInX0(G4double Lenght)
      { StepInX0=Lenght; } 
    inline void SetParameterisation(GVFlashShowerParameterisation &DP)
      { Parameterisation=&DP;}
    inline void SetHitMaker(GFlashHitMaker &Maker)
      { HMaker=&Maker; }
    inline void SetParticleBounds(GFlashParticleBounds &SpecificBound)
      { PBound =&SpecificBound; }
  
    // getting

    inline G4int GetFlagParamType()
      { return FlagParamType; }
    inline G4int GetFlagParticleContainment()
      { return FlagParticleContainment; }  
    inline G4double GetStepInX0()
      { return StepInX0; }

  public:  // without description

    // Gets ?  
    GFlashParticleBounds  *PBound;
    GVFlashShowerParameterisation *Parameterisation;  

  private:

    void ElectronDoIt(const G4FastTrack&, G4FastStep&);
    //  void GammaDoIt(const G4FastTrack&, G4FastStep&);
    //  void NeutrinoDoIt(const G4FastTrack&, G4FastStep&);
    G4bool CheckParticleDefAndContainment(const G4FastTrack &fastTrack);
    G4bool CheckContainment(const G4FastTrack &fastTrack);
  
  private:

    GFlashHitMaker *HMaker;  
    GflashEMShowerModelMessenger* Messenger;
  
    //Control Flags
    G4int FlagParamType;           ///0=no GFlash 1=only em showers parametrized
    G4int FlagParticleContainment; ///0=no check  ///1=only fully contained...
    G4double StepInX0;  
    G4double EnergyStop;
  
};
#endif

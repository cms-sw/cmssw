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
// $Id: G4MonopoleTransportation.hh,v 1.1 2010/07/29 23:05:19 sunanda Exp $
// GEANT4 tag $Name: CMSSW_6_2_0 $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// 
// ------------------------------------------------------------
//        GEANT 4  include file implementation
// ------------------------------------------------------------
//
// Class description:
//
// G4MonopoleTransportation is a process responsible for the transportation of 
// magnetic monopoles, i.e. the geometrical propagation encountering the 
// geometrical sub-volumes of the detectors.
// It is also tasked with part of updating the "safety".

// =======================================================================
// Created:  3 May 2010, J. Apostolakis, B. Bozsogi
// =======================================================================

#ifndef G4MonopoleTransportation_hh
#define G4MonopoleTransportation_hh 1

#include "G4VProcess.hh"
#include "G4FieldManager.hh"

#include "G4Navigator.hh"
#include "G4TransportationManager.hh"
#include "G4PropagatorInField.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ParticleChangeForTransport.hh"
//#include "G4MonopoleFieldSetup.hh"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"

class G4SafetyHelper; 
class G4Monopole;

#include <memory>

class G4MonopoleTransportation : public G4VProcess 
{
  // Concrete class that does the geometrical transport 

  public:  // with description

     G4MonopoleTransportation(const G4Monopole* p, sim::FieldBuilder *fieldBuilder, G4int verbosityLevel= 1);
     ~G4MonopoleTransportation(); 

     G4double      AlongStepGetPhysicalInteractionLength(
                             const G4Track& track,
                                   G4double  previousStepSize,
                                   G4double  currentMinimumStep, 
                                   G4double& currentSafety,
                                   G4GPILSelection* selection
                            );

     G4VParticleChange* AlongStepDoIt(
                             const G4Track& track,
                             const G4Step& stepData
                            );

     G4VParticleChange* PostStepDoIt(
                             const G4Track& track,
                             const G4Step&  stepData
                            );
       // Responsible for the relocation.

     G4double PostStepGetPhysicalInteractionLength(
                             const G4Track& ,
                             G4double   previousStepSize,
                             G4ForceCondition* pForceCond
                            );
       // Forces the PostStepDoIt action to be called, 
       // but does not limit the step.

     G4PropagatorInField* GetPropagatorInField();
     void SetPropagatorInField( G4PropagatorInField* pFieldPropagator);
       // Access/set the assistant class that Propagate in a Field.

     inline void   SetVerboseLevel( G4int verboseLevel );
     inline G4int  GetVerboseLevel() const;
       // Level of warnings regarding eg energy conservation
       // in field integration.

     inline G4double GetThresholdWarningEnergy() const; 
     inline G4double GetThresholdImportantEnergy() const; 
     inline G4int GetThresholdTrials() const; 

     inline void SetThresholdWarningEnergy( G4double newEnWarn ); 
     inline void SetThresholdImportantEnergy( G4double newEnImp ); 
     inline void SetThresholdTrials(G4int newMaxTrials ); 

     // Get/Set parameters for killing loopers: 
     //   Above 'important' energy a 'looping' particle in field will 
     //   *NOT* be abandoned, except after fThresholdTrials attempts.
     // Below Warning energy, no verbosity for looping particles is issued

     inline G4double GetMaxEnergyKilled() const; 
     inline G4double GetSumEnergyKilled() const;
     inline void ResetKilledStatistics( G4int report = 1);      
     // Statistics for tracks killed (currently due to looping in field)

     inline void EnableShortStepOptimisation(G4bool optimise=true); 
     // Whether short steps < safety will avoid to call Navigator (if field=0)

  public:  // without description

     G4double AtRestGetPhysicalInteractionLength(
                             const G4Track& ,
                             G4ForceCondition* 
                            ) { return -1.0; };
       // No operation in  AtRestDoIt.

     G4VParticleChange* AtRestDoIt(
                             const G4Track& ,
                             const G4Step&
                            ) {return 0;};
       // No operation in  AtRestDoIt.

  void StartTracking(G4Track* aTrack);
       // Reset state for new (potentially resumed) track 

  protected:

     G4bool               DoesGlobalFieldExist();
       // Checks whether a field exists for the "global" field manager.

  private:

     const G4Monopole* pParticleDef;

     //     G4MonopoleFieldSetup*  magSetup;

     sim::FieldBuilder *fFieldBuilder;

     G4Navigator*         fLinearNavigator;
     G4PropagatorInField* fFieldPropagator;
       // The Propagators used to transport the particle

     // G4FieldManager*      fGlobalFieldMgr;     // Used MagneticField CC
       // Field Manager for the whole Detector

     G4ThreeVector        fTransportEndPosition;
     G4ThreeVector        fTransportEndMomentumDir;
     G4double             fTransportEndKineticEnergy;
     G4ThreeVector        fTransportEndSpin;
     G4bool               fMomentumChanged;
     G4bool               fEnergyChanged;
     G4bool               fEndGlobalTimeComputed; 
     G4double             fCandidateEndGlobalTime;
       // The particle's state after this Step, Store for DoIt

     G4bool               fParticleIsLooping;

     G4TouchableHandle    fCurrentTouchableHandle;
     
     // G4bool         fFieldExists;
       // Whether a magnetic field exists ...
       // A data member for this is problematic: it is useful only if it
       // can be initialised and updated -- and a scheme is not yet possible.

     G4bool fGeometryLimitedStep;
       // Flag to determine whether a boundary was reached.

     G4ThreeVector  fPreviousSftOrigin;
     G4double       fPreviousSafety; 
       // Remember last safety origin & value.

     G4ParticleChangeForTransport fParticleChange;
       // New ParticleChange

     G4double endpointDistance;

  // Thresholds for looping particles: 
  // 
     G4double fThreshold_Warning_Energy;     //  Warn above this energy
     G4double fThreshold_Important_Energy;   //  Hesitate above this
     G4int    fThresholdTrials;              //    for this no of trials
       // Above 'important' energy a 'looping' particle in field will 
       //   *NOT* be abandoned, except after fThresholdTrials attempts.
     G4double fUnimportant_Energy;
       //  Below this energy, no verbosity for looping particles is issued

  // Counter for steps in which particle reports 'looping',
  //   if it is above 'Important' Energy 
     G4int    fNoLooperTrials; 
  // Statistics for tracks abandoned
     G4double fSumEnergyKilled;
     G4double fMaxEnergyKilled;

  // Whether to avoid calling G4Navigator for short step ( < safety)
  //   If using it, the safety estimate for endpoint will likely be smaller.
     G4bool   fShortStepOptimisation; 

     G4SafetyHelper* fpSafetyHelper;  // To pass it the safety value obtained

  // Verbosity 
     G4int    fVerboseLevel;
       // Verbosity level for warnings
       // eg about energy non-conservation in magnetic field.
};

#include "G4MonopoleTransportation.icc"

#endif  

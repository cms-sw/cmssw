//
// =======================================================================
//
// Class MonopoleTransportation
//
// Created:  3 May 2010, J. Apostolakis, B. Bozsogi 
//                       G4MonopoleTransportation class for
//                       Geant4 extended example "monopole"
//
// Adopted for CMSSW by V.Ivanchenko 30 April 2018
// from Geant4 global tag geant4-10-04-ref-03                   
//
// =======================================================================
// 
// Class description:
//
// G4MonopoleTransportation is a process responsible for the transportation of 
// magnetic monopoles, i.e. the geometrical propagation encountering the 
// geometrical sub-volumes of the detectors.
// It is also tasked with part of updating the "safety".
// For monopoles, uses a different equation of motion and ignores energy
// conservation. 
//

#ifndef SimG4Core_PhysicsLists_MonopoleTransportation_h
#define SimG4Core_PhysicsLists_MonopoleTransportation_h 1

#include "G4VProcess.hh"
#include "G4FieldManager.hh"

#include "G4Navigator.hh"
#include "G4TransportationManager.hh"
#include "G4PropagatorInField.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ParticleChangeForTransport.hh"
#include "CLHEP/Units/SystemOfUnits.h"

#include <memory>

class G4SafetyHelper; 
class Monopole;
class CMSFieldManager;

class MonopoleTransportation : public G4VProcess 
{
public: 

  MonopoleTransportation(const Monopole* p, G4int verbosityLevel= 1);
  ~MonopoleTransportation() override; 

  G4double AlongStepGetPhysicalInteractionLength(
                             const G4Track& track,
                                   G4double  previousStepSize,
                                   G4double  currentMinimumStep, 
                                   G4double& currentSafety,
                                   G4GPILSelection* selection
							 ) override;

  G4VParticleChange* AlongStepDoIt(
                             const G4Track& track,
                             const G4Step& stepData
					   ) override;

  G4VParticleChange* PostStepDoIt(
                             const G4Track& track,
                             const G4Step&  stepData
					  ) override;
  // Responsible for the relocation.

  G4double PostStepGetPhysicalInteractionLength(
                             const G4Track& ,
                             G4double   previousStepSize,
                             G4ForceCondition* pForceCond
							) override;
  // Forces the PostStepDoIt action to be called, 
  // but does not limit the step.

  G4PropagatorInField* GetPropagatorInField();
  void SetPropagatorInField( G4PropagatorInField* pFieldPropagator);
  // Access/set the assistant class that Propagate in a Field.

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
  void ResetKilledStatistics( G4int report = 1);      
  // Statistics for tracks killed (currently due to looping in field)

  inline void EnableShortStepOptimisation(G4bool optimise=true); 
  // Whether short steps < safety will avoid to call Navigator (if field=0)

  G4double AtRestGetPhysicalInteractionLength(const G4Track&,
                                              G4ForceCondition*) override;

  G4VParticleChange* AtRestDoIt(const G4Track&, const G4Step&) override;
  // No operation in  AtRestDoIt.

  void StartTracking(G4Track* aTrack) override;
  // Reset state for new (potentially resumed) track 

protected:

  G4bool               DoesGlobalFieldExist();
  // Checks whether a field exists for the "global" field manager.

private:

  const Monopole* fParticleDef;
  
  CMSFieldManager*     fieldMgrCMS;
    
  G4Navigator*         fLinearNavigator;
  G4PropagatorInField* fFieldPropagator;
  // The Propagators used to transport the particle

  G4ThreeVector        fTransportEndPosition;
  G4ThreeVector        fTransportEndMomentumDir;
  G4double             fTransportEndKineticEnergy;
  G4ThreeVector        fTransportEndSpin;
  G4bool               fMomentumChanged;

  G4bool               fEndGlobalTimeComputed; 
  G4double             fCandidateEndGlobalTime;
  // The particle's state after this Step, Store for DoIt

  G4bool               fParticleIsLooping;

  G4TouchableHandle    fCurrentTouchableHandle;

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

};

inline void
MonopoleTransportation::SetPropagatorInField( G4PropagatorInField* pFieldPropagator)
{
  fFieldPropagator= pFieldPropagator;
}

inline G4PropagatorInField* MonopoleTransportation::GetPropagatorInField()
{
  return fFieldPropagator;
}

inline G4bool MonopoleTransportation::DoesGlobalFieldExist()
{
  G4TransportationManager* transportMgr = 
    G4TransportationManager::GetTransportationManager();
  return transportMgr->GetFieldManager()->DoesFieldExist();
}

inline G4double MonopoleTransportation::GetThresholdWarningEnergy() const
{
  return fThreshold_Warning_Energy;
}
 
inline G4double MonopoleTransportation::GetThresholdImportantEnergy() const
{ 
  return fThreshold_Important_Energy;
} 

inline G4int MonopoleTransportation::GetThresholdTrials() const
{
  return fThresholdTrials;
}

inline void MonopoleTransportation::SetThresholdWarningEnergy( G4double newEnWarn )
{
  fThreshold_Warning_Energy= newEnWarn;
}

inline void MonopoleTransportation::SetThresholdImportantEnergy( G4double newEnImp )
{
  fThreshold_Important_Energy = newEnImp; 
}

inline void MonopoleTransportation::SetThresholdTrials(G4int newMaxTrials )
{
  fThresholdTrials = newMaxTrials; 
}

// Get parameters for killing loopers: 
//   Above 'important' energy a 'looping' particle in field will 
//   *NOT* be abandoned, except after fThresholdTrials attempts.
// Below Warning energy, no verbosity for looping particles is issued

inline G4double MonopoleTransportation::GetMaxEnergyKilled() const
{
  return fMaxEnergyKilled; 
}

inline G4double MonopoleTransportation::GetSumEnergyKilled() const
{
  return fSumEnergyKilled;
}

inline void 
MonopoleTransportation::EnableShortStepOptimisation(G4bool optimiseShortStep)
{ 
  fShortStepOptimisation=optimiseShortStep;
}

#endif  

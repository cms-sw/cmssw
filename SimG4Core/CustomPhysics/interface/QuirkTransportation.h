#ifndef SimG4Core_CustomPhysics_QuirkTransportation_H
#define SimG4Core_CustomPhysics_QuirkTransportation_H

// Transportation of quirks: each step swaps the field manager's chord finder
// for one driven by QuirkHyperbolaStepper and lets G4PropagatorInField find
// the boundary crossings.
// Ported from Athena Simulation/G4Extensions/Quirks,
// with the master/worker structure of MonopoleTransportation.

#include "G4VProcess.hh"
#include "G4FieldManager.hh"
#include "G4Navigator.hh"
#include "G4TransportationManager.hh"
#include "G4PropagatorInField.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ParticleChangeForTransport.hh"

class G4SafetyHelper;

class QuirkTransportation : public G4VProcess {
public:
  explicit QuirkTransportation(G4int verbosityLevel = 0, G4bool moveTinySteps = true, G4double crossingLength = 0.);
  ~QuirkTransportation() override;

  G4double AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                 G4double previousStepSize,
                                                 G4double currentMinimumStep,
                                                 G4double& currentSafety,
                                                 G4GPILSelection* selection) override;

  G4VParticleChange* AlongStepDoIt(const G4Track& track, const G4Step& stepData) override;

  G4double PostStepGetPhysicalInteractionLength(const G4Track&,
                                                G4double previousStepSize,
                                                G4ForceCondition* pForceCond) override;

  G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& stepData) override;

  G4double AtRestGetPhysicalInteractionLength(const G4Track&, G4ForceCondition*) override { return -1.0; }
  G4VParticleChange* AtRestDoIt(const G4Track&, const G4Step&) override { return nullptr; }

  void StartTracking(G4Track* aTrack) override;

  void SetThresholdTrials(G4int newMaxTrials) { fThresholdTrials = newMaxTrials; }

private:
  G4Navigator* fLinearNavigator{nullptr};
  G4PropagatorInField* fFieldPropagator{nullptr};
  G4SafetyHelper* fpSafetyHelper{nullptr};

  G4ThreeVector fTransportEndPosition;
  G4ThreeVector fTransportEndMomentumDir;
  G4double fTransportEndKineticEnergy{0.0};
  G4ThreeVector fTransportEndSpin;
  G4bool fMomentumChanged{false};
  G4double fCandidateEndGlobalTime{0.0};

  G4bool fParticleIsLooping{false};
  G4TouchableHandle fCurrentTouchableHandle;
  G4bool fGeometryLimitedStep{false};

  G4ThreeVector fPreviousSftOrigin;
  G4double fPreviousSafety{0.0};

  G4ParticleChangeForTransport fParticleChange;
  G4double fEndpointDistance{0.0};

  // looper killing
  G4double fThreshold_Warning_Energy;
  G4double fThreshold_Important_Energy;
  G4int fThresholdTrials;
  G4int fNoLooperTrials{0};
  G4double fSumEnergyKilled{0.0};
  G4double fMaxEnergyKilled{0.0};
  G4bool fMoveTinySteps;     // move the quirk on steps below the navigation tolerance
  G4double fCrossingLength;  // incoming string shorter than this: restart it (0: never)
};

#endif

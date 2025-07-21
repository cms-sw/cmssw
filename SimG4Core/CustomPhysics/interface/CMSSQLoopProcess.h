#ifndef CMSSQLoopProcess_h
#define CMSSQLoopProcess_h 1

#include "G4VContinuousProcess.hh"
#include "globals.hh"
#include "G4Track.hh"
#include "G4ParticleChange.hh"

class G4Step;
class G4ParticleDefinition;

class CMSSQLoopProcess : public G4VContinuousProcess {
public:
  CMSSQLoopProcess(const G4String& name = "SQLooper", G4ProcessType type = fUserDefined);
  ~CMSSQLoopProcess() override;

  G4VParticleChange* AlongStepDoIt(const G4Track&, const G4Step&) override;
  G4double AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                 G4double previousStepSize,
                                                 G4double currentMinimumStep,
                                                 G4double& proposedSafety,
                                                 G4GPILSelection* selection) override;
  void StartTracking(G4Track* aTrack) override;

  CMSSQLoopProcess(CMSSQLoopProcess&) = delete;
  CMSSQLoopProcess& operator=(const CMSSQLoopProcess& right) = delete;

protected:
  G4double GetContinuousStepLimit(const G4Track& track,
                                  G4double previousStepSize,
                                  G4double currentMinimumStep,
                                  G4double& currentSafety) override;

  G4ParticleChange* fParticleChange;

private:
  G4ThreeVector posini;
};

#endif

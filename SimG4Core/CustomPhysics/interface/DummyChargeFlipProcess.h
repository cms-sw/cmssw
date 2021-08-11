#ifndef SimG4Core_CustomPhysics_DummyChargeFlipProcess_h
#define SimG4Core_CustomPhysics_DummyChargeFlipProcess_h 1

#include "globals.hh"
#include "G4HadronicProcess.hh"
#include "G4ParticleChange.hh"

class DummyChargeFlipProcess : public G4HadronicProcess {
public:
  DummyChargeFlipProcess(const G4String& processName = "Dummy");

  ~DummyChargeFlipProcess() override;

  G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep) override;

  G4bool IsApplicable(const G4ParticleDefinition& aParticleType) override;

private:
  G4ParticleChange* fPartChange;
};
#endif

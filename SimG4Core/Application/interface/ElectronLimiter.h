#ifndef ElectronLimiter_h
#define ElectronLimiter_h 1

// V.Ivanchenko 2013/10/19
// step limiter and killer for e+,e-

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "globals.hh"
#include "G4VDiscreteProcess.hh"
#include "G4ParticleChangeForGamma.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4VEnergyLossProcess;

class ElectronLimiter : public G4VDiscreteProcess {
public:
  ElectronLimiter(const edm::ParameterSet& p);

  ~ElectronLimiter() override;

  void BuildPhysicsTable(const G4ParticleDefinition&) override;

  G4double PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                G4double previousStepSize,
                                                G4ForceCondition* condition) override;

  G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override;

  G4double GetMeanFreePath(const G4Track&, G4double, G4ForceCondition*) override;

  inline void SetRangeCheckFlag(G4bool);

  inline void SetFieldCheckFlag(G4bool);

private:
  G4ParticleChangeForGamma fParticleChange;
  G4VEnergyLossProcess* fIonisation;

  const G4ParticleDefinition* particle;

  G4double minStepLimit;

  G4bool rangeCheckFlag;
  G4bool fieldCheckFlag;
  G4bool killTrack;
};

inline void ElectronLimiter::SetRangeCheckFlag(G4bool val) { rangeCheckFlag = val; }

inline void ElectronLimiter::SetFieldCheckFlag(G4bool val) { fieldCheckFlag = val; }

#endif

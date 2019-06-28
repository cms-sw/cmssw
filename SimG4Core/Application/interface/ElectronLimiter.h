#ifndef ElectronLimiter_h
#define ElectronLimiter_h 1

// V.Ivanchenko 2013/10/19
// step limiter and killer for e+,e- and other charged particles

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "globals.hh"
#include "G4VEmProcess.hh"
#include "G4ParticleChangeForGamma.hh"
#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4Step;
class G4Track;
class G4Region;
class G4ParticleDefinition;
class G4VEnergyLossProcess;
class CMSTrackingCutModel;

class ElectronLimiter : public G4VEmProcess {
public:
  explicit ElectronLimiter(const edm::ParameterSet &, const G4ParticleDefinition *);

  ~ElectronLimiter() override;

  G4bool IsApplicable(const G4ParticleDefinition &) override;

  void InitialiseProcess(const G4ParticleDefinition *) override;

  void StartTracking(G4Track *) override;

  G4double PostStepGetPhysicalInteractionLength(const G4Track &track,
                                                G4double previousStepSize,
                                                G4ForceCondition *condition) override;

  G4VParticleChange *PostStepDoIt(const G4Track &, const G4Step &) override;

  void SetTrackingCutPerRegion(std::vector<const G4Region *> &,
                               std::vector<G4double> &,
                               std::vector<G4double> &,
                               std::vector<G4double> &);

  inline void SetRangeCheckFlag(G4bool);

  inline void SetFieldCheckFlag(G4bool);

private:
  G4VEnergyLossProcess *ionisation_;
  CMSTrackingCutModel *trcut_;
  const G4ParticleDefinition *particle_;

  std::vector<const G4Region *> regions_;
  std::vector<G4double> energyLimits_;
  std::vector<G4double> factors_;
  std::vector<G4double> rms_;

  G4double minStepLimit_;

  G4int nRegions_;
  G4bool rangeCheckFlag_;
  G4bool fieldCheckFlag_;
  G4bool killTrack_;
};

inline void ElectronLimiter::SetRangeCheckFlag(G4bool val) { rangeCheckFlag_ = val; }

inline void ElectronLimiter::SetFieldCheckFlag(G4bool val) { fieldCheckFlag_ = val; }

#endif

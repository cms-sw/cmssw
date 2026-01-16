#ifndef CMSSQLoopProcessDiscr_h
#define CMSSQLoopProcessDiscr_h 1

#include "G4VDiscreteProcess.hh"
#include "globals.hh"
#include "G4Track.hh"
#include "G4ParticleChange.hh"
#include "G4ParticleChangeForTransport.hh"
#include "CMSSQ.h"
#include "CMSAntiSQ.h"

class G4Step;
class G4ParticleDefinition;

class CMSSQLoopProcessDiscr : public G4VDiscreteProcess {
public:
  CMSSQLoopProcessDiscr(double mass, const G4String& name = "SQLooper", G4ProcessType type = fUserDefined);
  ~CMSSQLoopProcessDiscr() override;

public:
  G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override;
  G4double PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                G4double previousStepSize,
                                                G4ForceCondition* condition) override;
  G4double GetMeanFreePath(const G4Track&, G4double, G4ForceCondition*) override;
  void SetTimeLimit(G4double);
  void StartTracking(G4Track* aTrack) override;

  CMSSQLoopProcessDiscr(CMSSQLoopProcessDiscr&) = delete;
  CMSSQLoopProcessDiscr& operator=(const CMSSQLoopProcessDiscr& right) = delete;

protected:
  G4ParticleChange* fParticleChange;
  double GenMass;

private:
  G4ThreeVector posini;
  G4double globaltimeini;
};

#endif

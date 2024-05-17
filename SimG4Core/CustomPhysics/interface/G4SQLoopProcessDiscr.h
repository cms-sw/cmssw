#ifndef G4SQLoopProcessDiscr_h
#define G4SQLoopProcessDiscr_h 1

#include "G4VDiscreteProcess.hh"
#include "globals.hh"
#include "G4Track.hh"
#include "G4ParticleChange.hh"
#include "G4ParticleChangeForTransport.hh"
#include "G4SQ.h"
#include "G4AntiSQ.h"

class G4Step;
class G4ParticleDefinition;

class G4SQLoopProcessDiscr : public G4VDiscreteProcess {
public:
  G4SQLoopProcessDiscr(double mass, const G4String& name = "SQLooper", G4ProcessType type = fUserDefined);
  virtual ~G4SQLoopProcessDiscr();

public:
  virtual G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&);
  virtual G4double PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                        G4double previousStepSize,
                                                        G4ForceCondition* condition);
  virtual G4double GetMeanFreePath(const G4Track&, G4double, G4ForceCondition*);
  void SetTimeLimit(G4double);
  virtual void StartTracking(G4Track* aTrack);

private:
  G4SQLoopProcessDiscr(G4SQLoopProcessDiscr&);
  G4SQLoopProcessDiscr& operator=(const G4SQLoopProcessDiscr& right);

protected:
  //G4ParticleChangeForTransport* fParticleChange;
  G4ParticleChange* fParticleChange;
  double GenMass;

private:
  G4ThreeVector posini;
  G4double globaltimeini;
};

#endif

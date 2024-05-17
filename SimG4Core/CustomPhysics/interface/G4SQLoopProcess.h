#ifndef G4SQLoopProcess_h
#define G4SQLoopProcess_h 1

#include "G4VContinuousProcess.hh"
#include "globals.hh"
#include "G4Track.hh"
#include "G4ParticleChange.hh"


class G4Step;
class G4ParticleDefinition;


class G4SQLoopProcess : public G4VContinuousProcess {

  public:

    G4SQLoopProcess(const G4String& name = "SQLooper",
                          G4ProcessType type = fUserDefined);
    virtual ~G4SQLoopProcess();


  public:

    virtual G4VParticleChange* AlongStepDoIt(const G4Track&, const G4Step&);
    virtual G4double AlongStepGetPhysicalInteractionLength(const G4Track& track, G4double previousStepSize, G4double currentMinimumStep, G4double& proposedSafety, G4GPILSelection* selection);
    virtual void StartTracking(G4Track * aTrack);

  protected:

    virtual G4double GetContinuousStepLimit(const G4Track& track, G4double previousStepSize, G4double currentMinimumStep, G4double& currentSafety);
					
  private:

    G4SQLoopProcess(G4SQLoopProcess &);
    G4SQLoopProcess & operator=(const G4SQLoopProcess &right);

  protected:

    G4ParticleChange* fParticleChange;
  
  private:
 
    G4ThreeVector posini;
};


#endif

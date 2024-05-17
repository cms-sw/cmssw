#ifndef SimG4Core_CustomPhysics_G4SQInelasticProcess_h
#define SimG4Core_CustomPhysics_G4SQInelasticProcess_h 1
 
 
#include "G4HadronicProcess.hh"


class G4ParticleDefinition;


class G4SQInelasticProcess : public G4HadronicProcess
{

 public:

  G4SQInelasticProcess(double mass, const G4String& processName="SQInelastic");

  ~G4SQInelasticProcess();

  G4bool IsApplicable(const G4ParticleDefinition& aParticleType) override;

  // generic PostStepDoIt recommended for all derived classes
  virtual G4VParticleChange* PostStepDoIt(const G4Track& aTrack,
					  const G4Step& aStep);

  G4SQInelasticProcess& operator=(const G4SQInelasticProcess& right);
  G4SQInelasticProcess(const G4SQInelasticProcess&);

 protected:

  // Check the result for catastrophic energy non-conservation
  G4HadFinalState* CheckResult(const G4HadProjectile& thePro,
			       const G4Nucleus& targetNucleus,
			       G4HadFinalState* result);

 private:

  G4ParticleDefinition* theParticle;

  G4Nucleus targetNucleus;
  G4HadronicInteraction* theInteraction = nullptr;

};

#endif


#ifndef SIMAPPLICATION_APRIMEPHYSICS_H_
#define SIMAPPLICATION_APRIMEPHYSICS_H 1_

// Geant4
#include "G4VPhysicsConstructor.hh"

class APrimePhysics : public G4VPhysicsConstructor {
public:
  /**
       * Class constructor.
       * @param name The name of the physics.
       */
  APrimePhysics(double APMass, const G4String& scalefile, const G4double cxBias, const G4String& name = "APrime");

  /**
       * Class destructor.
       */
  ~APrimePhysics() override;

  /**
       * Construct particles.
       */
  void ConstructParticle() override;

  /**
       * Construct the process.
       */
  void ConstructProcess() override;

private:
  /**
       * Definition of the APrime particle.
       */
  G4ParticleDefinition* aprimeDef_;
  double apmass;
  G4String mgfile;
  G4double biasFactor;
};

#endif

#ifndef SimG4Core_CustomPhysics_CMSSQInelasticProcess_h
#define SimG4Core_CustomPhysics_CMSSQInelasticProcess_h 1

#include "G4HadronicProcess.hh"

class G4ParticleDefinition;

class CMSSQInelasticProcess : public G4HadronicProcess {

public:
  CMSSQInelasticProcess(double mass, const G4String& processName = "SQInelastic");

  ~CMSSQInelasticProcess() = default;

  CMSSQInelasticProcess& operator=(const CMSSQInelasticProcess& right);
  CMSSQInelasticProcess(const CMSSQInelasticProcess&);

private:

  G4ParticleDefinition* theParticle;

//  G4Nucleus targetNucleus;
//  G4HadronicInteraction* theInteraction = nullptr;
};

#endif

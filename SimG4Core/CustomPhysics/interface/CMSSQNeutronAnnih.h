
#ifndef CMSSQNeutronAnnih_h
#define CMSSQNeutronAnnih_h 1

#include "globals.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadProjectile.hh"
#include "G4Nucleus.hh"
#include "G4IonTable.hh"

class G4ParticleDefinition;

class CMSSQNeutronAnnih : public G4HadronicInteraction {
public:
  CMSSQNeutronAnnih(double mass);

  ~CMSSQNeutronAnnih() override;

  G4double momDistr(G4double x_in);

  G4HadFinalState* ApplyYourself(const G4HadProjectile& aTrack, G4Nucleus& targetNucleus) override;

private:
  G4ParticleDefinition* theSQ;
  G4ParticleDefinition* theK0S;
  G4ParticleDefinition* theAntiL;
  G4ParticleDefinition* theProton;
};

#endif

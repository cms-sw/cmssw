
#ifndef G4SQNeutronAnnih_h
#define G4SQNeutronAnnih_h 1

#include "globals.hh"
#include "G4HadronicInteraction.hh"
#include "G4HadProjectile.hh"
#include "G4Nucleus.hh"
#include "G4IonTable.hh"

class G4ParticleDefinition;


class G4SQNeutronAnnih : public G4HadronicInteraction {

  public:

    G4SQNeutronAnnih(double mass);

    virtual ~G4SQNeutronAnnih();

    G4double momDistr(G4double x_in);

    virtual G4HadFinalState * ApplyYourself(
                   const G4HadProjectile & aTrack,
                   G4Nucleus & targetNucleus);

  private:

    G4ParticleDefinition* theSQ;
    G4ParticleDefinition* theK0S;
    G4ParticleDefinition* theAntiL;
    G4ParticleDefinition* theProton;

};

#endif


#ifndef G4SQ_h
#define G4SQ_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"

// ######################################################################
// ###                         SEXAQUARK                              ###
// ######################################################################

class G4SQ : public G4ParticleDefinition {
private:
  static G4SQ* theInstance;
  G4SQ() {}
  ~G4SQ() {}

public:
  static G4SQ* Definition(double mass);
  //    static G4SQ* SQDefinition(double mass);
  static G4SQ* SQ(double mass);
};

#endif


#ifndef CMSSQ_h
#define CMSSQ_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"

// ######################################################################
// ###                         SEXAQUARK                              ###
// ######################################################################

class CMSSQ : public G4ParticleDefinition {
private:
  static CMSSQ* theInstance;
  CMSSQ() {}
  ~CMSSQ() {}

public:
  static CMSSQ* Definition(double mass);
  static CMSSQ* SQ(double mass);
};

#endif

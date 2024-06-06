#ifndef CMSAntiSQ_h
#define CMSAntiSQ_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"

// ######################################################################
// ###                      ANTI-SEXAQUARK                            ###
// ######################################################################

class CMSAntiSQ : public G4ParticleDefinition {
private:
  static CMSAntiSQ* theInstance;
  CMSAntiSQ() {}
  ~CMSAntiSQ() {}

public:
  static CMSAntiSQ* Definition(double mass);
  static CMSAntiSQ* AntiSQ(double mass);
};

#endif

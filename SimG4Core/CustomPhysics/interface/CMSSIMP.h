#ifndef SimG4Core_CustomPhysics_CMSSIMP_H
#define SimG4Core_CustomPhysics_CMSSIMP_H

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"

class CMSSIMP : public G4ParticleDefinition {
private:
  static CMSSIMP* theInstance;
  CMSSIMP() {}
  ~CMSSIMP() override {}

public:
  static CMSSIMP* Definition(double mass);
  static CMSSIMP* SIMPDefinition(double mass);
  static CMSSIMP* SIMP();
};

#endif

#ifndef SimG4Core_CustomPhysics_CMSAntiSIMP_H
#define SimG4Core_CustomPhysics_CMSAntiSIMP_H

#include "globals.hh"
#include "G4ios.hh"
#include "G4ParticleDefinition.hh"

class CMSAntiSIMP : public G4ParticleDefinition {
private:
  static CMSAntiSIMP* theInstance;
  CMSAntiSIMP() {}
  ~CMSAntiSIMP() override {}

public:
  static CMSAntiSIMP* Definition(double mass);
  static CMSAntiSIMP* AntiSIMPDefinition(double mass);
  static CMSAntiSIMP* AntiSIMP();
};

#endif

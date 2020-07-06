#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsLPM : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsLPM(G4int ver);
  ~CMSEmStandardPhysicsLPM() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int verbose;
};

#endif

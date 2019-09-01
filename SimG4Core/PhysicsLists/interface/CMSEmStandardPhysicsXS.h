#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsXS_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsXS_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsXS : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsXS(G4int ver);
  ~CMSEmStandardPhysicsXS() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int verbose;
};

#endif

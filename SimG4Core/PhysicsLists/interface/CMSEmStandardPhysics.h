#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysics : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysics(G4int ver, G4int ntr, G4double wEn, G4double iEn);
  ~CMSEmStandardPhysics() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  const G4int         verbose_, ntrials_;
  const G4double      wEnergy_, iEnergy_;
};

#endif







#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsLPM : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysicsLPM(G4int ver, G4int ntr, G4double wEn, G4double iEn);
  ~CMSEmStandardPhysicsLPM() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  const G4int         verbose_, ntrials_;
  const G4double      wEnergy_, iEnergy_;
};

#endif







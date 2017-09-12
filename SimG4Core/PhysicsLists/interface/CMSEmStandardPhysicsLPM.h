#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsLPM : public G4VPhysicsConstructor {

public: 
  CMSEmStandardPhysicsLPM(G4int ver);
  virtual ~CMSEmStandardPhysicsLPM();

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:
  G4int               verbose;
};

#endif







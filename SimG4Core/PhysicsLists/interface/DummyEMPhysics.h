#ifndef SimG4Core_PhysicsLists_DummyEMPhysics_h
#define SimG4Core_PhysicsLists_DummyEMPhysics_h

#include "G4VPhysicsConstructor.hh"

class DummyEMPhysics : public G4VPhysicsConstructor {

public: 
  DummyEMPhysics(const std::string name = "dummyEM");
  virtual ~DummyEMPhysics();
  virtual void ConstructParticle();
  virtual void ConstructProcess();
};

#endif


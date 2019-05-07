#ifndef SimG4Core_PhysicsLists_DummyPhysics_H
#define SimG4Core_PhysicsLists_DummyPhysics_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

class DummyPhysics : public PhysicsList {

public:
  DummyPhysics(const edm::ParameterSet &);
  ~DummyPhysics() override = default;
};

#endif

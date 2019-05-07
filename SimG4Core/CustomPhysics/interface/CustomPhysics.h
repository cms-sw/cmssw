#ifndef SimG4Core_CustomPhysics_CustomPhysics_H
#define SimG4Core_CustomPhysics_CustomPhysics_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

class CustomPhysics : public PhysicsList {
public:
  CustomPhysics(const edm::ParameterSet &p);
};

#endif

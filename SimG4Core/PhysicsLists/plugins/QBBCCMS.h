#ifndef SimG4Core_PhysicsLists_QBBCCMS_H
#define SimG4Core_PhysicsLists_QBBCCMS_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

class QBBCCMS : public PhysicsList {

public:
  QBBCCMS(const edm::ParameterSet &p);
};

#endif

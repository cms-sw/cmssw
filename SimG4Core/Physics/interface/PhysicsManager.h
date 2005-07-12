#ifndef SimG4Core_PhysicsManager_H
#define SimG4Core_PhysicsManager_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SealKernel/Component.h"

class PhysicsManager : public seal::Component
{
public:
    PhysicsManager(seal::Context * context,edm::ParameterSet & pSet);
};

#endif

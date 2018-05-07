#ifndef SimG4Core_CustomPhysics_CustomPhysics_H
#define SimG4Core_CustomPhysics_CustomPhysics_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class CustomPhysics : public PhysicsList
{
public:
    CustomPhysics(const edm::ParameterSet & p);
};
 
#endif

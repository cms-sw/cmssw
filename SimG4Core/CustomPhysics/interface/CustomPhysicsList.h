#ifndef SimG4Core_CustomPhysicsList_H
#define SimG4Core_CustomPhysicsList_H
 
#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
class CustomPhysicsList : public PhysicsList
{
public:
    CustomPhysicsList(const edm::ParameterSet & p);
};
 
#endif

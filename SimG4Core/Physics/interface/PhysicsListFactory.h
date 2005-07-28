#ifndef SimG4Core_PhysicsListFactory_H
#define SimG4Core_PhysicsListFactory_H

#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "SealKernel/Component.h"
#include "PluginManager/PluginFactory.h"

class PhysicsListFactory 
    : public seal::PluginFactory<
    PhysicsList * (seal::Context *,const edm::ParameterSet & p) >
{
public:
    virtual ~PhysicsListFactory();
    static PhysicsListFactory * get(); 
private:
    static PhysicsListFactory s_instance;
    PhysicsListFactory();
};

#endif

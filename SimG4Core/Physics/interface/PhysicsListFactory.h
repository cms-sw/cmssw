#ifndef SimG4Core_PhysicsListFactory_H
#define SimG4Core_PhysicsListFactory_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "SimG4Core/Physics/interface/PhysicsListMaker.h"

#include "SealKernel/Component.h"
#include "PluginManager/PluginFactory.h"

class PhysicsListFactory 
    : public seal::PluginFactory<
    PhysicsListMakerBase *() >
{
public:
    virtual ~PhysicsListFactory();
    static PhysicsListFactory * get(); 
private:
    static PhysicsListFactory s_instance;
    PhysicsListFactory();
};
//NOTE: the prefix "SimG4Core/Physics/" is there for 'backwards compatability
// and should eventually be removed (which will require changes to config files)
#define DEFINE_PHYSICSLIST(type) \
  DEFINE_SEAL_PLUGIN(PhysicsListFactory, PhysicsListMaker<type>,"SimG4Core/Physics/" #type)

#endif

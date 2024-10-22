#ifndef SimG4Core_PhysicsListFactory_H
#define SimG4Core_PhysicsListFactory_H

#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "SimG4Core/Physics/interface/PhysicsListMaker.h"

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<PhysicsListMakerBase *()> PhysicsListFactory;
//NOTE: the prefix "SimG4Core/Physics/" is there for 'backwards compatability
// and should eventually be removed (which will require changes to config files)
#define DEFINE_PHYSICSLIST(type) \
  DEFINE_EDM_PLUGIN(PhysicsListFactory, PhysicsListMaker<type>, "SimG4Core/Physics/" #type)

#endif

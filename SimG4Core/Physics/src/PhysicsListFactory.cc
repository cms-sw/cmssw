#include "SimG4Core/Physics/interface/PhysicsListFactory.h"

PhysicsListFactory PhysicsListFactory::s_instance;

PhysicsListFactory::PhysicsListFactory()
    : seal::PluginFactory<PhysicsList * 
(seal::Context *,const edm::ParameterSet &)>("Sim Physics Plugins")
{}

PhysicsListFactory::~PhysicsListFactory() {}

PhysicsListFactory * PhysicsListFactory::get() { return & s_instance; }

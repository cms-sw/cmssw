#include "SimG4Core/Physics/interface/PhysicsListFactory.h"

PhysicsListFactory PhysicsListFactory::s_instance;

PhysicsListFactory::PhysicsListFactory()
    : seal::PluginFactory<PhysicsListMakerBase * ()>("CMS Simulation PhysicsListFactory")
{}

PhysicsListFactory::~PhysicsListFactory() {}

PhysicsListFactory * PhysicsListFactory::get() { return & s_instance; }

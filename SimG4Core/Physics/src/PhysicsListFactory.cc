//____________________________________________________________________
// File: PhysicsListFactory.cc
//____________________________________________________________________
//
// Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
// Update: 2005-07-05 12:32:32+0200
// Revision: $Id$
//
//--------------------------------------------------------------------
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
 
PhysicsListFactory PhysicsListFactory::s_instance;
 
PhysicsListFactory::PhysicsListFactory()
    : seal::PluginFactory<PhysicsList* (seal::Context *)>("Sim Physics Plugins")
{}
 
PhysicsListFactory::~PhysicsListFactory()
{}
 
PhysicsListFactory * PhysicsListFactory::get()
{
    return &s_instance;
}

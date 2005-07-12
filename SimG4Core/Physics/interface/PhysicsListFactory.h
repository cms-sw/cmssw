//____________________________________________________________________
// File: PhysicsListFactory.h
//____________________________________________________________________
//
// Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
// Update: 2005-07-05 12:25:28+0200
// Revision: $Id$
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsListFactory_H
#define SimG4Core_PhysicsListFactory_H

#include <iosfwd>
 
#include "SealKernel/Component.h"
#include "PluginManager/PluginFactory.h"
 
class PhysicsList;
 
class PhysicsListFactory : public seal::PluginFactory<PhysicsList * (seal::Context *)>
{
public:
    virtual ~PhysicsListFactory();
    static PhysicsListFactory * get();
private:
    static PhysicsListFactory s_instance;
    PhysicsListFactory();
};
 
#endif

//____________________________________________________________________
// File: PhysicsList.h
//____________________________________________________________________
//
// Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
// Update: 2005-07-05 12:23:47+0200
// Revision: $Id$
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsList_H
#define SimG4Core_PhysicsList_H

#include <iosfwd>
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SealKernel/Component.h"

#include "G4VModularPhysicsList.hh"

class PhysicsList : public G4VModularPhysicsList, 
		    public seal::Component
{
    DECLARE_SEAL_COMPONENT;
public:
    PhysicsList(seal::Context * context, 
		const edm::ParameterSet & pSet);
    virtual ~PhysicsList();
    virtual void SetCuts();
private:
    seal::Context * m_context;
    edm::ParameterSet m_paramSet;
};

#endif

//____________________________________________________________________
// File: PhysicsList.cc
//____________________________________________________________________
//
// Author: Shaun ASHBY <Shaun.Ashby@cern.ch>
// Update: 2005-07-05 12:31:45+0200
// Revision: $Id$
//
//--------------------------------------------------------------------
#include "SimG4Core/Physics/interface/PhysicsList.h"

DEFINE_SEAL_COMPONENT (PhysicsList, "SimG4Core/PhysicsList");
 
PhysicsList::PhysicsList(seal::Context * context, const edm::ParameterSet & p)
    : Component(context, classContextKey()),
      m_context(context), m_paramSet(p)
{}
 
PhysicsList::~PhysicsList() {}

void PhysicsList::SetCuts() 
{ 
    SetDefaultCutValue(m_paramSet.getParameter<double>("DefaultCut")*cm);
    SetCutsWithDefault();
    if (m_paramSet.getParameter<int>("Verbosity")>0) 
	G4VUserPhysicsList::DumpCutValuesTable();
}


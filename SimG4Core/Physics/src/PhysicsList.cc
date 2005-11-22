#include "SimG4Core/Physics/interface/PhysicsList.h"

PhysicsList::PhysicsList(const edm::ParameterSet & p) 
    : G4VModularPhysicsList(), m_pPhysics(p) {}
 
PhysicsList::~PhysicsList() {}

void PhysicsList::SetCuts() 
{ 
    SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*cm);
    SetCutsWithDefault();
    if (m_pPhysics.getParameter<int>("Verbosity") > 1) 
	G4VUserPhysicsList::DumpCutValuesTable();
}


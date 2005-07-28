#include "SimG4Core/Physics/interface/PhysicsList.h"

DEFINE_SEAL_COMPONENT (PhysicsList, "SimG4Core/Physics/PhysicsList");
 
PhysicsList::PhysicsList(seal::Context * c, const edm::ParameterSet & p) 
    : G4VModularPhysicsList(), Component(c, classContextKey()),
      m_context(c), m_pPhysics(p) {}
 
PhysicsList::~PhysicsList() {}

void PhysicsList::SetCuts() 
{ 
    SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*cm);
    SetCutsWithDefault();
    if (m_pPhysics.getParameter<int>("Verbosity") > 1) 
	G4VUserPhysicsList::DumpCutValuesTable();
}


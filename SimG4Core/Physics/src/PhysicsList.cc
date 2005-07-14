#include "SimG4Core/Physics/interface/PhysicsList.h"

PhysicsList::PhysicsList() : G4VModularPhysicsList()
{}
 
PhysicsList::~PhysicsList() {}

void PhysicsList::SetCuts() 
{ 
    SetDefaultCutValue(10.*cm);
    SetCutsWithDefault();
    G4VUserPhysicsList::DumpCutValuesTable();
}


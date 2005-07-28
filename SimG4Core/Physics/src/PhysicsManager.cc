#include "SimG4Core/Physics/interface/PhysicsManager.h"
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"
#include "SimG4Core/Physics/interface/PhysicsList.h"

#include "G4RunManagerKernel.hh"

PhysicsManager::PhysicsManager(seal::Context * c,edm::ParameterSet & p)
    : Component(c,"SimG4Core/PhysicsManager")
{
//     edm::ParameterSet physicsSet
//           = p.getParameter<edm::ParameterSet>("Physics");
//     seal::Handle<PhysicsList> physics = PhysicsListFactory::get()->create
//         (physicsSet.getParameter<std::string> ("type"), context);
//     ASSERT (physics); 
//     G4RunManagerKernel * kernel = G4RunManagerKernel::GetRunManagerKernel();
//     kernel->SetPhysics(physics.get());
}

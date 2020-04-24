#include "SimG4Core/PhysicsLists/interface/DummyEMPhysics.h"

#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4eIonisation.hh"
#include "G4MuIonisation.hh"

DummyEMPhysics::DummyEMPhysics(const std::string name) 
  : G4VPhysicsConstructor(name) {}

DummyEMPhysics::~DummyEMPhysics() {}

void DummyEMPhysics::ConstructParticle() {
  G4Electron::ElectronDefinition();     
  G4MuonMinus::MuonMinusDefinition();
}

void DummyEMPhysics::ConstructProcess() {
  G4ProcessManager * man = nullptr;
  man = G4Electron::Electron()->GetProcessManager();
  man->AddProcess(new G4eIonisation,	  -1, 2,2);
  man = G4MuonMinus::MuonMinus()->GetProcessManager();
  man->AddProcess(new G4MuIonisation,	  -1, 2,2);
}

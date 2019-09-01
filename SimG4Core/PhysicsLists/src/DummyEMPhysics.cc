#include "SimG4Core/PhysicsLists/interface/DummyEMPhysics.h"
#include "G4EmParameters.hh"
#include "G4ParticleTable.hh"

#include "G4ParticleDefinition.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4eMultipleScattering.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4LeptonConstructor.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"

#include "G4SystemOfUnits.hh"

DummyEMPhysics::DummyEMPhysics(G4int ver) : G4VPhysicsConstructor("CMSEmGeantV"), verbose(ver) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(verbose);
  param->SetApplyCuts(true);
  param->SetStepFunction(0.8, 1 * CLHEP::mm);
  param->SetLossFluctuations(false);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  SetPhysicsType(bElectromagnetic);
}

void DummyEMPhysics::ConstructParticle() {
  // gamma
  G4Gamma::Gamma();

  // leptons
  G4Electron::Electron();
  G4Positron::Positron();

  G4LeptonConstructor pLeptonConstructor;
  pLeptonConstructor.ConstructParticle();
}

void DummyEMPhysics::ConstructProcess() {
  if (verbose > 0) {
    G4cout << "### " << GetPhysicsName() << " Construct Processes " << G4endl;
  }

  // This EM builder takes GeantV variant of physics

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  G4ParticleDefinition* particle = G4Gamma::Gamma();

  ph->RegisterProcess(new G4PhotoElectricEffect(), particle);
  ph->RegisterProcess(new G4ComptonScattering(), particle);
  ph->RegisterProcess(new G4GammaConversion(), particle);

  particle = G4Electron::Electron();

  ph->RegisterProcess(new G4eMultipleScattering(), particle);
  ph->RegisterProcess(new G4eIonisation(), particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);

  particle = G4Positron::Positron();

  ph->RegisterProcess(new G4eMultipleScattering(), particle);
  ph->RegisterProcess(new G4eIonisation(), particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);
  ph->RegisterProcess(new G4eplusAnnihilation(), particle);
}

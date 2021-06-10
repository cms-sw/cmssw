#include "SimG4Core/PhysicsLists/interface/CMSEmNoDeltaRay.h"
#include "SimG4Core/PhysicsLists/interface/EmParticleList.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4EmParameters.hh"
#include "G4ParticleTable.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4LossTableManager.hh"
#include "G4RegionStore.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4UrbanMscModel.hh"

#include "G4hMultipleScattering.hh"
#include "G4eMultipleScattering.hh"
#include "G4MscStepLimitType.hh"

#include "G4hhIonisation.hh"

#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"

#include "G4hBremsstrahlung.hh"
#include "G4hPairProduction.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonMinus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4GenericIon.hh"

#include "G4EmBuilder.hh"
#include "G4BuilderType.hh"
#include "G4SystemOfUnits.hh"

CMSEmNoDeltaRay::CMSEmNoDeltaRay(const G4String& name, G4int ver, const std::string& reg)
    : G4VPhysicsConstructor(name), region(reg) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(ver);
  param->SetApplyCuts(true);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  SetPhysicsType(bElectromagnetic);
}

CMSEmNoDeltaRay::~CMSEmNoDeltaRay() {}

void CMSEmNoDeltaRay::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmNoDeltaRay::ConstructProcess() {
  // Add standard EM Processes
  G4Region* reg = nullptr;
  if (region != " ") {
    G4RegionStore* regStore = G4RegionStore::GetInstance();
    reg = regStore->GetRegion(region, false);
  }

  G4ParticleTable* table = G4ParticleTable::GetParticleTable();
  EmParticleList emList;
  for (const auto& particleName : emList.PartNames()) {
    G4ParticleDefinition* particle = table->FindParticle(particleName);
    G4ProcessManager* pmanager = particle->GetProcessManager();
    if (verboseLevel > 1)
      edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " instantiates for " << particleName;

    if (particleName == "gamma") {
      pmanager->AddDiscreteProcess(new G4PhotoElectricEffect);
      pmanager->AddDiscreteProcess(new G4ComptonScattering);
      pmanager->AddDiscreteProcess(new G4GammaConversion);

    } else if (particleName == "e-") {
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      msc->SetStepLimitType(fMinimal);
      if (reg != nullptr) {
        G4UrbanMscModel* msc_el = new G4UrbanMscModel();
        msc_el->SetRangeFactor(0.04);
        msc->AddEmModel(0, msc_el, reg);
      }
      pmanager->AddProcess(msc, -1, 1, -1);
      pmanager->AddProcess(new G4hhIonisation, -1, 2, 1);
      pmanager->AddProcess(new G4eBremsstrahlung, -1, -3, 2);

    } else if (particleName == "e+") {
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      msc->SetStepLimitType(fMinimal);
      if (reg != nullptr) {
        G4UrbanMscModel* msc_pos = new G4UrbanMscModel();
        msc_pos->SetRangeFactor(0.04);
        msc->AddEmModel(0, msc_pos, reg);
      }
      pmanager->AddProcess(msc, -1, 1, -1);
      pmanager->AddProcess(new G4hhIonisation, -1, 2, -1);
      pmanager->AddProcess(new G4eBremsstrahlung, -1, -3, 1);
      pmanager->AddProcess(new G4eplusAnnihilation, 0, -1, 2);

    } else if (particleName == "mu+" || particleName == "mu-") {
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, -1);
      pmanager->AddProcess(new G4hhIonisation, -1, 2, -1);
      pmanager->AddProcess(new G4MuBremsstrahlung, -1, -3, 1);
      pmanager->AddProcess(new G4MuPairProduction, -1, -4, 2);

    } else if (particleName == "alpha" || particleName == "He3" || particleName == "GenericIon") {
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, -1);
      pmanager->AddProcess(new G4hhIonisation, -1, 2, -1);

    } else if (particleName == "pi+" || particleName == "kaon+" || particleName == "kaon-" ||
               particleName == "proton" || particleName == "pi-") {
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, -1);
      pmanager->AddProcess(new G4hhIonisation, -1, 2, -1);
      pmanager->AddProcess(new G4hBremsstrahlung(), -1, -3, 1);
      pmanager->AddProcess(new G4hPairProduction(), -1, -4, 2);

    } else if (particle->GetPDGCharge() != 0.0) {
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, -1);
      pmanager->AddProcess(new G4hhIonisation, -1, 2, -1);
    }
  }
}

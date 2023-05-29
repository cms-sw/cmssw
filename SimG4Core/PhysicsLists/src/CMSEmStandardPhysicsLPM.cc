#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsLPM.h"
#include "SimG4Core/PhysicsLists/interface/EmParticleList.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4EmParameters.hh"
#include "G4ParticleTable.hh"

#include "G4ParticleDefinition.hh"
#include "G4LossTableManager.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4hMultipleScattering.hh"
#include "G4eMultipleScattering.hh"
#include "G4MuMultipleScattering.hh"
#include "G4CoulombScattering.hh"
#include "G4eCoulombScatteringModel.hh"
#include "G4WentzelVIModel.hh"
#include "G4UrbanMscModel.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"
#include "G4UAtomicDeexcitation.hh"

#include "G4MuIonisation.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"

#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"
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

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4GammaGeneralProcess.hh"
#include "G4EmBuilder.hh"

#include "G4SystemOfUnits.hh"

CMSEmStandardPhysicsLPM::CMSEmStandardPhysicsLPM(G4int ver, const edm::ParameterSet& p)
    : G4VPhysicsConstructor("CMSEmStandard_emm") {
  SetVerboseLevel(ver);
  // EM parameters specific for this EM physics configuration
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(ver);
  param->SetApplyCuts(true);
  param->SetStepFunction(0.8, 1 * CLHEP::mm);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  SetPhysicsType(bElectromagnetic);
  double tcut = p.getParameter<double>("G4TrackingCut") * CLHEP::MeV;
  param->SetLowestElectronEnergy(tcut);
  param->SetLowestMuHadEnergy(tcut);
}

void CMSEmStandardPhysicsLPM::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmStandardPhysicsLPM::ConstructProcess() {
  if (verboseLevel > 1) {
    edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " Construct Processes";
  }

  // This EM builder takes default models of Geant4 10 EMV.
  // Multiple scattering by Urban for all particles
  // except e+e- below 100 MeV for which the Urban model is used

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  G4LossTableManager* man = G4LossTableManager::Instance();

  // muon & hadron bremsstrahlung and pair production
  G4MuBremsstrahlung* mub = nullptr;
  G4MuPairProduction* mup = nullptr;
  G4hBremsstrahlung* pib = nullptr;
  G4hPairProduction* pip = nullptr;
  G4hBremsstrahlung* kb = nullptr;
  G4hPairProduction* kp = nullptr;
  G4hBremsstrahlung* pb = nullptr;
  G4hPairProduction* pp = nullptr;

  // muon & hadron multiple scattering
  G4MuMultipleScattering* mumsc = nullptr;
  G4hMultipleScattering* pimsc = nullptr;
  G4hMultipleScattering* kmsc = nullptr;
  G4hMultipleScattering* pmsc = nullptr;
  G4hMultipleScattering* hmsc = nullptr;

  // muon and hadron single scattering
  G4CoulombScattering* muss = nullptr;
  G4CoulombScattering* piss = nullptr;
  G4CoulombScattering* kss = nullptr;

  // high energy limit for e+- scattering models and bremsstrahlung
  G4double highEnergyLimit = 100 * MeV;

  G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion", false);
  G4Region* bRegion = G4RegionStore::GetInstance()->GetRegion("HGCalRegion", false);
  if (verboseLevel > 1) {
    edm::LogVerbatim("PhysicsList") << "CMSEmStandardPhysicsLPM: HcalRegion " << aRegion << "; HGCalRegion " << bRegion;
  }
  G4ParticleTable* table = G4ParticleTable::GetParticleTable();
  EmParticleList emList;
  for (const auto& particleName : emList.PartNames()) {
    G4ParticleDefinition* particle = table->FindParticle(particleName);

    if (particleName == "gamma") {
      G4PhotoElectricEffect* pee = new G4PhotoElectricEffect();

      if (G4EmParameters::Instance()->GeneralProcessActive()) {
        G4GammaGeneralProcess* sp = new G4GammaGeneralProcess();
        sp->AddEmProcess(pee);
        sp->AddEmProcess(new G4ComptonScattering());
        sp->AddEmProcess(new G4GammaConversion());
        man->SetGammaGeneralProcess(sp);
        ph->RegisterProcess(sp, particle);

      } else {
        ph->RegisterProcess(pee, particle);
        ph->RegisterProcess(new G4ComptonScattering(), particle);
        ph->RegisterProcess(new G4GammaConversion(), particle);
      }

    } else if (particleName == "e-") {
      G4eIonisation* eioni = new G4eIonisation();

      G4eMultipleScattering* msc = new G4eMultipleScattering;
      G4UrbanMscModel* msc1 = new G4UrbanMscModel();
      G4WentzelVIModel* msc2 = new G4WentzelVIModel();
      msc1->SetHighEnergyLimit(highEnergyLimit);
      msc2->SetLowEnergyLimit(highEnergyLimit);
      msc->SetEmModel(msc1);
      msc->SetEmModel(msc2);

      // e-/e+ msc for HCAL and HGCAL using the Urban model
      if (nullptr != aRegion || nullptr != bRegion) {
        G4UrbanMscModel* msc3 = new G4UrbanMscModel();
        msc3->SetHighEnergyLimit(highEnergyLimit);
        msc3->SetLocked(true);

        if (nullptr != aRegion) {
          msc->AddEmModel(-1, msc3, aRegion);
        }
        if (nullptr != bRegion) {
          msc->AddEmModel(-1, msc3, bRegion);
        }
      }

      G4eCoulombScatteringModel* ssm = new G4eCoulombScatteringModel();
      G4CoulombScattering* ss = new G4CoulombScattering();
      ss->SetEmModel(ssm);
      ss->SetMinKinEnergy(highEnergyLimit);
      ssm->SetLowEnergyLimit(highEnergyLimit);
      ssm->SetActivationLowEnergyLimit(highEnergyLimit);

      ph->RegisterProcess(msc, particle);
      ph->RegisterProcess(eioni, particle);
      ph->RegisterProcess(new G4eBremsstrahlung(), particle);
      ph->RegisterProcess(ss, particle);

    } else if (particleName == "e+") {
      G4eIonisation* eioni = new G4eIonisation();

      G4eMultipleScattering* msc = new G4eMultipleScattering;
      G4UrbanMscModel* msc1 = new G4UrbanMscModel();
      G4WentzelVIModel* msc2 = new G4WentzelVIModel();
      msc1->SetHighEnergyLimit(highEnergyLimit);
      msc2->SetLowEnergyLimit(highEnergyLimit);
      msc->SetEmModel(msc1);
      msc->SetEmModel(msc2);

      // e-/e+ msc for HCAL and HGCAL using the Urban model
      if (nullptr != aRegion || nullptr != bRegion) {
        G4UrbanMscModel* msc3 = new G4UrbanMscModel();
        msc3->SetHighEnergyLimit(highEnergyLimit);
        msc3->SetLocked(true);

        if (nullptr != aRegion) {
          msc->AddEmModel(-1, msc3, aRegion);
        }
        if (nullptr != bRegion) {
          msc->AddEmModel(-1, msc3, bRegion);
        }
      }

      G4eCoulombScatteringModel* ssm = new G4eCoulombScatteringModel();
      G4CoulombScattering* ss = new G4CoulombScattering();
      ss->SetEmModel(ssm);
      ss->SetMinKinEnergy(highEnergyLimit);
      ssm->SetLowEnergyLimit(highEnergyLimit);
      ssm->SetActivationLowEnergyLimit(highEnergyLimit);

      ph->RegisterProcess(msc, particle);
      ph->RegisterProcess(eioni, particle);
      ph->RegisterProcess(new G4eBremsstrahlung(), particle);
      ph->RegisterProcess(new G4eplusAnnihilation(), particle);
      ph->RegisterProcess(ss, particle);

    } else if (particleName == "mu+" || particleName == "mu-") {
      if (nullptr == mub) {
        mub = new G4MuBremsstrahlung();
        mup = new G4MuPairProduction();
        mumsc = new G4MuMultipleScattering();
        mumsc->SetEmModel(new G4WentzelVIModel());
        muss = new G4CoulombScattering();
      }
      ph->RegisterProcess(mumsc, particle);
      ph->RegisterProcess(new G4MuIonisation(), particle);
      ph->RegisterProcess(mub, particle);
      ph->RegisterProcess(mup, particle);
      ph->RegisterProcess(muss, particle);

    } else if (particleName == "alpha" || particleName == "He3") {
      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4ionIonisation(), particle);

    } else if (particleName == "GenericIon") {
      if (nullptr == hmsc) {
        hmsc = new G4hMultipleScattering("ionmsc");
      }
      ph->RegisterProcess(hmsc, particle);
      ph->RegisterProcess(new G4ionIonisation(), particle);

    } else if (particleName == "pi+" || particleName == "pi-") {
      if (nullptr == pib) {
        pib = new G4hBremsstrahlung();
        pip = new G4hPairProduction();
        pimsc = new G4hMultipleScattering();
        pimsc->SetEmModel(new G4WentzelVIModel());
        piss = new G4CoulombScattering();
      }
      ph->RegisterProcess(pimsc, particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
      ph->RegisterProcess(pib, particle);
      ph->RegisterProcess(pip, particle);
      ph->RegisterProcess(piss, particle);

    } else if (particleName == "kaon+" || particleName == "kaon-") {
      if (nullptr == kb) {
        kb = new G4hBremsstrahlung();
        kp = new G4hPairProduction();
        kmsc = new G4hMultipleScattering();
        kmsc->SetEmModel(new G4WentzelVIModel());
        kss = new G4CoulombScattering();
      }
      ph->RegisterProcess(kmsc, particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
      ph->RegisterProcess(kb, particle);
      ph->RegisterProcess(kp, particle);
      ph->RegisterProcess(kss, particle);

    } else if (particleName == "proton" || particleName == "anti_proton") {
      if (nullptr == pb) {
        pb = new G4hBremsstrahlung();
        pp = new G4hPairProduction();
      }
      pmsc = new G4hMultipleScattering();
      pmsc->SetEmModel(new G4WentzelVIModel());

      ph->RegisterProcess(pmsc, particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
      ph->RegisterProcess(pb, particle);
      ph->RegisterProcess(pp, particle);
      ph->RegisterProcess(new G4CoulombScattering(), particle);

    } else if (particle->GetPDGCharge() != 0.0) {
      if (nullptr == hmsc) {
        hmsc = new G4hMultipleScattering("ionmsc");
      }
      ph->RegisterProcess(hmsc, particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
    }
  }
  edm::LogVerbatim("PhysicsList") << "CMSEmStandardPhysicsLPM: EM physics is instantiated";
}

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
#include "G4TauMinus.hh"
#include "G4TauPlus.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"
#include "G4KaonPlus.hh"
#include "G4KaonMinus.hh"
#include "G4BMesonMinus.hh"
#include "G4BMesonPlus.hh"
#include "G4DMesonMinus.hh"
#include "G4DMesonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4SigmaMinus.hh"
#include "G4AntiSigmaMinus.hh"
#include "G4SigmaPlus.hh"
#include "G4AntiSigmaPlus.hh"
#include "G4XiMinus.hh"
#include "G4AntiXiMinus.hh"
#include "G4OmegaMinus.hh"
#include "G4AntiOmegaMinus.hh"
#include "G4LambdacPlus.hh"
#include "G4AntiLambdacPlus.hh"
#include "G4XicPlus.hh"
#include "G4AntiXicPlus.hh"
#include "G4Deuteron.hh"
#include "G4Triton.hh"
#include "G4He3.hh"
#include "G4Alpha.hh"
#include "G4GenericIon.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"

#include "G4SystemOfUnits.hh"

CMSEmStandardPhysicsLPM::CMSEmStandardPhysicsLPM(G4int ver) : G4VPhysicsConstructor("CMSEmStandard_emm"), verbose(ver) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(verbose);
  param->SetApplyCuts(true);
  param->SetStepFunction(0.8, 1 * CLHEP::mm);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  SetPhysicsType(bElectromagnetic);
}

CMSEmStandardPhysicsLPM::~CMSEmStandardPhysicsLPM() {}

void CMSEmStandardPhysicsLPM::ConstructParticle() {
  // gamma
  G4Gamma::Gamma();

  // leptons
  G4Electron::Electron();
  G4Positron::Positron();
  G4MuonPlus::MuonPlus();
  G4MuonMinus::MuonMinus();
  G4TauMinus::TauMinusDefinition();
  G4TauPlus::TauPlusDefinition();

  // mesons
  G4PionPlus::PionPlusDefinition();
  G4PionMinus::PionMinusDefinition();
  G4KaonPlus::KaonPlusDefinition();
  G4KaonMinus::KaonMinusDefinition();
  G4DMesonMinus::DMesonMinusDefinition();
  G4DMesonPlus::DMesonPlusDefinition();
  G4BMesonMinus::BMesonMinusDefinition();
  G4BMesonPlus::BMesonPlusDefinition();

  // barions
  G4Proton::Proton();
  G4AntiProton::AntiProton();
  G4SigmaMinus::SigmaMinusDefinition();
  G4AntiSigmaMinus::AntiSigmaMinusDefinition();
  G4SigmaPlus::SigmaPlusDefinition();
  G4AntiSigmaPlus::AntiSigmaPlusDefinition();
  G4XiMinus::XiMinusDefinition();
  G4AntiXiMinus::AntiXiMinusDefinition();
  G4OmegaMinus::OmegaMinusDefinition();
  G4AntiOmegaMinus::AntiOmegaMinusDefinition();
  G4LambdacPlus::LambdacPlusDefinition();
  G4AntiLambdacPlus::AntiLambdacPlusDefinition();
  G4XicPlus::XicPlusDefinition();
  G4AntiXicPlus::AntiXicPlusDefinition();

  // ions
  G4Deuteron::Deuteron();
  G4Triton::Triton();
  G4He3::He3();
  G4Alpha::Alpha();
  G4GenericIon::GenericIonDefinition();
}

void CMSEmStandardPhysicsLPM::ConstructProcess() {
  if (verbose > 1) {
    edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " Construct Processes";
  }

  // This EM builder takes default models of Geant4 10 EMV.
  // Multiple scattering by Urban for all particles
  // except e+e- below 100 MeV for which the Urban model is used

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

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
  if (verbose > 1) {
    edm::LogVerbatim("PhysicsList") << "CMSEmStandardPhysicsLPM: HcalRegion " << aRegion << "; HGCalRegion " << bRegion;
  }
  G4ParticleTable* table = G4ParticleTable::GetParticleTable();
  EmParticleList emList;
  for (const auto& particleName : emList.PartNames()) {
    G4ParticleDefinition* particle = table->FindParticle(particleName);

    if (particleName == "gamma") {
      ph->RegisterProcess(new G4PhotoElectricEffect(), particle);
      ph->RegisterProcess(new G4ComptonScattering(), particle);
      ph->RegisterProcess(new G4GammaConversion(), particle);

    } else if (particleName == "e-") {
      G4eIonisation* eioni = new G4eIonisation();

      G4eMultipleScattering* msc = new G4eMultipleScattering;
      G4UrbanMscModel* msc1 = new G4UrbanMscModel();
      G4WentzelVIModel* msc2 = new G4WentzelVIModel();
      G4UrbanMscModel* msc3 = new G4UrbanMscModel();
      msc1->SetHighEnergyLimit(highEnergyLimit);
      msc2->SetLowEnergyLimit(highEnergyLimit);
      msc3->SetHighEnergyLimit(highEnergyLimit);
      msc3->SetLocked(true);
      msc->SetEmModel(msc1);
      msc->SetEmModel(msc2);
      if (aRegion) {
        msc->AddEmModel(-1, msc3, aRegion);
      }
      if (bRegion) {
        msc->AddEmModel(-1, msc3, bRegion);
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
      G4UrbanMscModel* msc3 = new G4UrbanMscModel();
      msc1->SetHighEnergyLimit(highEnergyLimit);
      msc2->SetLowEnergyLimit(highEnergyLimit);
      msc3->SetHighEnergyLimit(highEnergyLimit);
      msc3->SetLocked(true);
      msc->SetEmModel(msc1);
      msc->SetEmModel(msc2);
      if (aRegion) {
        msc->AddEmModel(-1, msc3, aRegion);
      }
      if (bRegion) {
        msc->AddEmModel(-1, msc3, bRegion);
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

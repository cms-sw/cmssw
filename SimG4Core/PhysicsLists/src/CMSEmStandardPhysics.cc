#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSHepEmTrackingManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4EmParameters.hh"
#include "G4EmBuilder.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"

#include "G4MscStepLimitType.hh"

#include "G4eMultipleScattering.hh"
#include "G4hMultipleScattering.hh"
#include "G4eCoulombScatteringModel.hh"
#include "G4CoulombScattering.hh"
#include "G4WentzelVIModel.hh"
#include "G4UrbanMscModel.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"

#include "G4ParticleTable.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4GammaGeneralProcess.hh"
#include "G4LossTableManager.hh"

#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1110
#include "G4ProcessManager.hh"
#include "G4TransportationWithMsc.hh"
#endif

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include <string>

CMSEmStandardPhysics::CMSEmStandardPhysics(G4int ver, const edm::ParameterSet& p)
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
  param->SetFluo(false);
  SetPhysicsType(bElectromagnetic);
  fRangeFactor = p.getParameter<double>("G4MscRangeFactor");
  fGeomFactor = p.getParameter<double>("G4MscGeomFactor");
  fSafetyFactor = p.getParameter<double>("G4MscSafetyFactor");
  fLambdaLimit = p.getParameter<double>("G4MscLambdaLimit") * CLHEP::mm;
  std::string msc = p.getParameter<std::string>("G4MscStepLimit");
  fStepLimitType = fUseSafety;
  if (msc == "UseSafetyPlus") {
    fStepLimitType = fUseSafetyPlus;
  }
  if (msc == "Minimal") {
    fStepLimitType = fMinimal;
  }
  double tcut = p.getParameter<double>("G4TrackingCut") * CLHEP::MeV;
  param->SetLowestElectronEnergy(tcut);
  param->SetLowestMuHadEnergy(tcut);
  fG4HepEmActive = p.getParameter<bool>("G4HepEmActive");
}

void CMSEmStandardPhysics::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmStandardPhysics::ConstructProcess() {
  if (verboseLevel > 0) {
    edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " Construct EM Processes";
  }

  // This EM builder takes default models of Geant4 10 EMV.
  // Multiple scattering by WentzelVI for all particles except:
  //   a) e+e- below 100 MeV for which the Urban model is used
  //   b) ions for which Urban model is used
  G4EmBuilder::PrepareEMPhysics();

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  // processes used by several particles
  G4hMultipleScattering* hmsc = new G4hMultipleScattering("ionmsc");
  G4NuclearStopping* pnuc(nullptr);

  // high energy limit for e+- scattering models
  auto param = G4EmParameters::Instance();
  G4double highEnergyLimit = param->MscEnergyLimit();

  const G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion", false);
  const G4Region* bRegion = G4RegionStore::GetInstance()->GetRegion("HGCalRegion", false);

  // Add gamma EM Processes
  G4ParticleDefinition* particle = G4Gamma::Gamma();

  G4PhotoElectricEffect* pee = new G4PhotoElectricEffect();

  if (param->GeneralProcessActive()) {
    G4GammaGeneralProcess* sp = new G4GammaGeneralProcess();
    sp->AddEmProcess(pee);
    sp->AddEmProcess(new G4ComptonScattering());
    sp->AddEmProcess(new G4GammaConversion());
    G4LossTableManager::Instance()->SetGammaGeneralProcess(sp);
    ph->RegisterProcess(sp, particle);

  } else {
    ph->RegisterProcess(pee, particle);
    ph->RegisterProcess(new G4ComptonScattering(), particle);
    ph->RegisterProcess(new G4GammaConversion(), particle);
  }

  // e-
  particle = G4Electron::Electron();

  G4eIonisation* eioni = new G4eIonisation();

  G4UrbanMscModel* msc1 = new G4UrbanMscModel();
  G4WentzelVIModel* msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);

  // e-/e+ msc for HCAL and HGCAL using the Urban model
  G4UrbanMscModel* msc3 = nullptr;
  if (nullptr != aRegion || nullptr != bRegion) {
    msc3 = new G4UrbanMscModel();
    msc3->SetHighEnergyLimit(highEnergyLimit);
    msc3->SetRangeFactor(fRangeFactor);
    msc3->SetGeomFactor(fGeomFactor);
    msc3->SetSafetyFactor(fSafetyFactor);
    msc3->SetLambdaLimit(fLambdaLimit);
    msc3->SetStepLimitType(fStepLimitType);
    msc3->SetLocked(true);
  }

#if G4VERSION_NUMBER >= 1110
  G4TransportationWithMscType transportationWithMsc = param->TransportationWithMsc();
  if (transportationWithMsc != G4TransportationWithMscType::fDisabled) {
    // Remove default G4Transportation and replace with G4TransportationWithMsc.
    G4ProcessManager* procManager = particle->GetProcessManager();
    G4VProcess* removed = procManager->RemoveProcess(0);
    if (removed->GetProcessName() != "Transportation") {
      G4Exception("CMSEmStandardPhysics::ConstructProcess",
                  "em0050",
                  FatalException,
                  "replaced process is not G4Transportation!");
    }
    G4TransportationWithMsc* transportWithMsc =
        new G4TransportationWithMsc(G4TransportationWithMsc::ScatteringType::MultipleScattering);
    if (transportationWithMsc == G4TransportationWithMscType::fMultipleSteps) {
      transportWithMsc->SetMultipleSteps(true);
    }
    transportWithMsc->AddMscModel(msc1);
    transportWithMsc->AddMscModel(msc2);
    if (nullptr != aRegion) {
      transportWithMsc->AddMscModel(msc3, -1, aRegion);
    }
    if (nullptr != bRegion) {
      transportWithMsc->AddMscModel(msc3, -1, bRegion);
    }
    procManager->AddProcess(transportWithMsc, -1, 0, 0);
  } else
#endif
  {
    // Multiple scattering is registered as a separate process
    G4eMultipleScattering* msc = new G4eMultipleScattering;
    msc->SetEmModel(msc1);
    msc->SetEmModel(msc2);
    if (nullptr != aRegion) {
      msc->AddEmModel(-1, msc3, aRegion);
    }
    if (nullptr != bRegion) {
      msc->AddEmModel(-1, msc3, bRegion);
    }
    ph->RegisterProcess(msc, particle);
  }

  // single scattering
  G4eCoulombScatteringModel* ssm = new G4eCoulombScatteringModel();
  G4CoulombScattering* ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  ph->RegisterProcess(eioni, particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);
  ph->RegisterProcess(ss, particle);

  // e+
  particle = G4Positron::Positron();
  eioni = new G4eIonisation();

  msc1 = new G4UrbanMscModel();
  msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);

  // e-/e+ msc for HCAL and HGCAL using the Urban model
  if (nullptr != aRegion || nullptr != bRegion) {
    msc3 = new G4UrbanMscModel();
    msc3->SetHighEnergyLimit(highEnergyLimit);
    msc3->SetRangeFactor(fRangeFactor);
    msc3->SetGeomFactor(fGeomFactor);
    msc3->SetSafetyFactor(fSafetyFactor);
    msc3->SetLambdaLimit(fLambdaLimit);
    msc3->SetStepLimitType(fStepLimitType);
    msc3->SetLocked(true);
  }

#if G4VERSION_NUMBER >= 1110
  if (transportationWithMsc != G4TransportationWithMscType::fDisabled) {
    G4ProcessManager* procManager = particle->GetProcessManager();
    // Remove default G4Transportation and replace with G4TransportationWithMsc.
    G4VProcess* removed = procManager->RemoveProcess(0);
    if (removed->GetProcessName() != "Transportation") {
      G4Exception("CMSEmStandardPhysics::ConstructProcess",
                  "em0050",
                  FatalException,
                  "replaced process is not G4Transportation!");
    }
    G4TransportationWithMsc* transportWithMsc =
        new G4TransportationWithMsc(G4TransportationWithMsc::ScatteringType::MultipleScattering);
    if (transportationWithMsc == G4TransportationWithMscType::fMultipleSteps) {
      transportWithMsc->SetMultipleSteps(true);
    }
    transportWithMsc->AddMscModel(msc1);
    transportWithMsc->AddMscModel(msc2);
    if (nullptr != aRegion) {
      transportWithMsc->AddMscModel(msc3, -1, aRegion);
    }
    if (nullptr != bRegion) {
      transportWithMsc->AddMscModel(msc3, -1, bRegion);
    }
    procManager->AddProcess(transportWithMsc, -1, 0, 0);
  } else
#endif
  {
    // Register as a separate process.
    G4eMultipleScattering* msc = new G4eMultipleScattering;
    msc->SetEmModel(msc1);
    msc->SetEmModel(msc2);
    if (nullptr != aRegion) {
      msc->AddEmModel(-1, msc3, aRegion);
    }
    if (nullptr != bRegion) {
      msc->AddEmModel(-1, msc3, bRegion);
    }
    ph->RegisterProcess(msc, particle);
  }

  // single scattering
  ssm = new G4eCoulombScatteringModel();
  ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  ph->RegisterProcess(eioni, particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);
  ph->RegisterProcess(new G4eplusAnnihilation(), particle);
  ph->RegisterProcess(ss, particle);

#if G4VERSION_NUMBER >= 1110
  if (fG4HepEmActive) {
    auto* hepEmTM = new CMSHepEmTrackingManager(highEnergyLimit);
    G4Electron::Electron()->SetTrackingManager(hepEmTM);
    G4Positron::Positron()->SetTrackingManager(hepEmTM);
  }
#endif

  // generic ion
  particle = G4GenericIon::GenericIon();
  G4ionIonisation* ionIoni = new G4ionIonisation();
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);

  // muons, hadrons ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);
}

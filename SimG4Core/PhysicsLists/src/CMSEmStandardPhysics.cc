#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
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

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include <string>

CMSEmStandardPhysics::CMSEmStandardPhysics(G4int ver, const edm::ParameterSet& p)
    : G4VPhysicsConstructor("CMSEmStandard_emm") {
  SetVerboseLevel(ver);
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
}

CMSEmStandardPhysics::~CMSEmStandardPhysics() {}

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
  G4double highEnergyLimit = G4EmParameters::Instance()->MscEnergyLimit();

  const G4Region* aRegion = G4RegionStore::GetInstance()->GetRegion("HcalRegion", false);
  const G4Region* bRegion = G4RegionStore::GetInstance()->GetRegion("HGCalRegion", false);

  // Add gamma EM Processes
  G4ParticleDefinition* particle = G4Gamma::Gamma();

  G4PhotoElectricEffect* pee = new G4PhotoElectricEffect();

  if (G4EmParameters::Instance()->GeneralProcessActive()) {
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
    msc3->SetRangeFactor(fRangeFactor);
    msc3->SetGeomFactor(fGeomFactor);
    msc3->SetSafetyFactor(fSafetyFactor);
    msc3->SetLambdaLimit(fLambdaLimit);
    msc3->SetStepLimitType(fStepLimitType);
    msc3->SetLocked(true);

    if (nullptr != aRegion) {
      msc->AddEmModel(-1, msc3, aRegion);
    }
    if (nullptr != bRegion) {
      msc->AddEmModel(-1, msc3, bRegion);
    }
  }

  // single scattering
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

  // e+
  particle = G4Positron::Positron();
  eioni = new G4eIonisation();

  msc = new G4eMultipleScattering();
  msc1 = new G4UrbanMscModel();
  msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);
  msc->SetEmModel(msc1);
  msc->SetEmModel(msc2);

  // e-/e+ msc for HCAL and HGCAL using the Urban model
  if (nullptr != aRegion || nullptr != bRegion) {
    G4UrbanMscModel* msc3 = new G4UrbanMscModel();
    msc3->SetHighEnergyLimit(highEnergyLimit);
    msc3->SetRangeFactor(fRangeFactor);
    msc3->SetGeomFactor(fGeomFactor);
    msc3->SetSafetyFactor(fSafetyFactor);
    msc3->SetLambdaLimit(fLambdaLimit);
    msc3->SetStepLimitType(fStepLimitType);
    msc3->SetLocked(true);

    if (nullptr != aRegion) {
      msc->AddEmModel(-1, msc3, aRegion);
    }
    if (nullptr != bRegion) {
      msc->AddEmModel(-1, msc3, bRegion);
    }
  }

  // single scattering
  ssm = new G4eCoulombScatteringModel();
  ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  ph->RegisterProcess(msc, particle);
  ph->RegisterProcess(eioni, particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);
  ph->RegisterProcess(new G4eplusAnnihilation(), particle);
  ph->RegisterProcess(ss, particle);

  // generic ion
  particle = G4GenericIon::GenericIon();
  G4ionIonisation* ionIoni = new G4ionIonisation();
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);

  // muons, hadrons ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);
}

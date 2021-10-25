#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsXS.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4LossTableManager.hh"
#include "G4EmParameters.hh"
#include "G4EmBuilder.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4RayleighScattering.hh"
#include "G4PEEffectFluoModel.hh"
#include "G4KleinNishinaModel.hh"
#include "G4LowEPComptonModel.hh"
#include "G4BetheHeitler5DModel.hh"
#include "G4LivermorePhotoElectricModel.hh"

#include "G4eMultipleScattering.hh"
#include "G4hMultipleScattering.hh"
#include "G4MscStepLimitType.hh"
#include "G4UrbanMscModel.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4DummyModel.hh"
#include "G4WentzelVIModel.hh"
#include "G4CoulombScattering.hh"
#include "G4eCoulombScatteringModel.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4Generator2BS.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4ePairProduction.hh"
#include "G4UniversalFluctuation.hh"

#include "G4eplusAnnihilation.hh"

#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4GammaGeneralProcess.hh"

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4GammaGeneralProcess.hh"

#include "G4SystemOfUnits.hh"

CMSEmStandardPhysicsXS::CMSEmStandardPhysicsXS(G4int ver, const edm::ParameterSet& p)
    : G4VPhysicsConstructor("CMSEmStandard_emn") {
  SetVerboseLevel(ver);
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(ver);
  param->SetApplyCuts(true);
  param->SetMinEnergy(100 * CLHEP::eV);
  param->SetNumberOfBinsPerDecade(20);
  param->SetStepFunction(0.8, 1 * CLHEP::mm);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  param->SetFluo(true);
  param->SetUseMottCorrection(true);  // use Mott-correction for e-/e+ msc gs
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

CMSEmStandardPhysicsXS::~CMSEmStandardPhysicsXS() {}

void CMSEmStandardPhysicsXS::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmStandardPhysicsXS::ConstructProcess() {
  if (verboseLevel > 0) {
    edm::LogVerbatim("PhysicsList") << "### " << GetPhysicsName() << " Construct Processes";
  }

  // This EM builder takes default models of Geant4 10 EMV.
  // Multiple scattering by Urban for all particles
  // except e+e- below 100 MeV for which the Urban model is used
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

  // Photoelectric
  G4PhotoElectricEffect* pe = new G4PhotoElectricEffect();
  G4VEmModel* theLivermorePEModel = new G4LivermorePhotoElectricModel();
  pe->SetEmModel(theLivermorePEModel);

  // Compton scattering
  G4ComptonScattering* cs = new G4ComptonScattering;
  cs->SetEmModel(new G4KleinNishinaModel());

  // Gamma conversion
  G4GammaConversion* gc = new G4GammaConversion();
  G4VEmModel* conv = new G4BetheHeitler5DModel();
  gc->SetEmModel(conv);

  if (G4EmParameters::Instance()->GeneralProcessActive()) {
    G4GammaGeneralProcess* sp = new G4GammaGeneralProcess();
    sp->AddEmProcess(pe);
    sp->AddEmProcess(cs);
    sp->AddEmProcess(gc);
    sp->AddEmProcess(new G4RayleighScattering());
    G4LossTableManager::Instance()->SetGammaGeneralProcess(sp);
    ph->RegisterProcess(sp, particle);
  } else {
    ph->RegisterProcess(pe, particle);
    ph->RegisterProcess(cs, particle);
    ph->RegisterProcess(gc, particle);
    ph->RegisterProcess(new G4RayleighScattering(), particle);
  }

  // e-
  particle = G4Electron::Electron();

  // multiple scattering
  G4eMultipleScattering* msc = new G4eMultipleScattering();
  G4UrbanMscModel* msc1 = new G4UrbanMscModel();
  G4WentzelVIModel* msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);
  msc->SetEmModel(msc1);
  msc->SetEmModel(msc2);

  // msc for HCAL using the Urban model
  if (nullptr != aRegion) {
    G4UrbanMscModel* msc4 = new G4UrbanMscModel();
    msc4->SetHighEnergyLimit(highEnergyLimit);
    msc4->SetRangeFactor(fRangeFactor);
    msc4->SetGeomFactor(fGeomFactor);
    msc4->SetSafetyFactor(fSafetyFactor);
    msc4->SetLambdaLimit(fLambdaLimit);
    msc4->SetStepLimitType(fStepLimitType);
    msc4->SetLocked(true);
    msc->AddEmModel(-1, msc4, aRegion);
  }

  // msc GS with Mott-correction
  if (nullptr != bRegion) {
    G4GoudsmitSaundersonMscModel* msc3 = new G4GoudsmitSaundersonMscModel();
    msc3->SetHighEnergyLimit(highEnergyLimit);
    msc3->SetRangeFactor(0.08);
    msc3->SetSkin(3.);
    msc3->SetStepLimitType(fUseSafetyPlus);
    msc3->SetLocked(true);
    msc->AddEmModel(-1, msc3, bRegion);
  }

  // single scattering
  G4eCoulombScatteringModel* ssm = new G4eCoulombScatteringModel();
  G4CoulombScattering* ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  // ionisation
  G4eIonisation* eioni = new G4eIonisation();

  // bremsstrahlung
  G4eBremsstrahlung* brem = new G4eBremsstrahlung();
  G4SeltzerBergerModel* br1 = new G4SeltzerBergerModel();
  G4eBremsstrahlungRelModel* br2 = new G4eBremsstrahlungRelModel();
  br1->SetAngularDistribution(new G4Generator2BS());
  br2->SetAngularDistribution(new G4Generator2BS());
  brem->SetEmModel(br1);
  brem->SetEmModel(br2);
  br1->SetHighEnergyLimit(CLHEP::GeV);

  G4ePairProduction* ee = new G4ePairProduction();

  // register processes
  ph->RegisterProcess(msc, particle);
  ph->RegisterProcess(eioni, particle);
  ph->RegisterProcess(brem, particle);
  ph->RegisterProcess(ee, particle);
  ph->RegisterProcess(ss, particle);

  // e+
  particle = G4Positron::Positron();

  // multiple scattering
  msc = new G4eMultipleScattering();
  msc1 = new G4UrbanMscModel();
  msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);
  msc->SetEmModel(msc1);
  msc->SetEmModel(msc2);

  // msc for HCAL using the Urban model
  if (nullptr != aRegion) {
    G4UrbanMscModel* msc4 = new G4UrbanMscModel();
    msc4->SetHighEnergyLimit(highEnergyLimit);
    msc4->SetRangeFactor(fRangeFactor);
    msc4->SetGeomFactor(fGeomFactor);
    msc4->SetSafetyFactor(fSafetyFactor);
    msc4->SetLambdaLimit(fLambdaLimit);
    msc4->SetStepLimitType(fStepLimitType);
    msc4->SetLocked(true);
    msc->AddEmModel(-1, msc4, aRegion);
  }

  // msc GS with Mott-correction
  if (nullptr != bRegion) {
    G4GoudsmitSaundersonMscModel* msc3 = new G4GoudsmitSaundersonMscModel();
    msc3->SetHighEnergyLimit(highEnergyLimit);
    msc3->SetRangeFactor(0.08);
    msc3->SetSkin(3.);
    msc3->SetStepLimitType(fUseSafetyPlus);
    msc3->SetLocked(true);
    msc->AddEmModel(-1, msc3, bRegion);
  }

  // single scattering
  ssm = new G4eCoulombScatteringModel();
  ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  // ionisation
  eioni = new G4eIonisation();

  // bremsstrahlung
  brem = new G4eBremsstrahlung();
  br1 = new G4SeltzerBergerModel();
  br2 = new G4eBremsstrahlungRelModel();
  br1->SetAngularDistribution(new G4Generator2BS());
  br2->SetAngularDistribution(new G4Generator2BS());
  brem->SetEmModel(br1);
  brem->SetEmModel(br2);
  br1->SetHighEnergyLimit(CLHEP::GeV);

  // register processes
  ph->RegisterProcess(msc, particle);
  ph->RegisterProcess(eioni, particle);
  ph->RegisterProcess(brem, particle);
  ph->RegisterProcess(ee, particle);
  ph->RegisterProcess(new G4eplusAnnihilation(), particle);
  ph->RegisterProcess(ss, particle);

  // generic ion
  particle = G4GenericIon::GenericIon();
  G4ionIonisation* ionIoni = new G4ionIonisation();
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);

  // muons, hadrons, ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);
}

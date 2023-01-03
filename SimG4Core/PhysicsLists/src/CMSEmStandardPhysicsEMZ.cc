#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsEMZ.h"
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

#include "G4IonParametrisedLossModel.hh"
#include "G4NuclearStopping.hh"
#include "G4ePairProduction.hh"
#include "G4LivermoreIonisationModel.hh"
#include "G4PenelopeIonisationModel.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4GammaGeneralProcess.hh"

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4GammaGeneralProcess.hh"

#include "G4SystemOfUnits.hh"

CMSEmStandardPhysicsEMZ::CMSEmStandardPhysicsEMZ(G4int ver, const edm::ParameterSet& p)
    : G4VPhysicsConstructor("CMSEmStandard_emz") {
  SetVerboseLevel(ver);
  // EM parameters specific for this EM physics configuration
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(ver);
  param->SetMinEnergy(100 * CLHEP::eV);
  param->SetLowestElectronEnergy(100 * CLHEP::eV);
  param->SetNumberOfBinsPerDecade(20);
  param->ActivateAngularGeneratorForIonisation(true);
  param->SetStepFunction(0.2, 10 * CLHEP::um);
  param->SetStepFunctionMuHad(0.1, 50 * CLHEP::um);
  param->SetStepFunctionLightIons(0.1, 20 * CLHEP::um);
  param->SetStepFunctionIons(0.1, 1 * CLHEP::um);
  param->SetUseMottCorrection(true);           // use Mott-correction for e-/e+ msc gs
  param->SetMscStepLimitType(fUseSafetyPlus);  // for e-/e+ msc gs
  param->SetMscSkin(3);                        // error-free stepping for e-/e+ msc gs
  param->SetMscRangeFactor(0.08);              // error-free stepping for e-/e+ msc gs
  param->SetMuHadLateralDisplacement(true);
  param->SetFluo(true);
  param->SetUseICRU90Data(true);
  param->SetMaxNIELEnergy(1 * CLHEP::MeV);
  double tcut = p.getParameter<double>("G4TrackingCut") * CLHEP::MeV;
  param->SetLowestElectronEnergy(tcut);
  param->SetLowestMuHadEnergy(tcut);
  SetPhysicsType(bElectromagnetic);
}

void CMSEmStandardPhysicsEMZ::ConstructParticle() {
  // minimal set of particles for EM physics
  G4EmBuilder::ConstructMinimalEmSet();
}

void CMSEmStandardPhysicsEMZ::ConstructProcess() {
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
  G4double nielEnergyLimit = G4EmParameters::Instance()->MaxNIELEnergy();
  G4NuclearStopping* pnuc = nullptr;
  if (nielEnergyLimit > 0.0) {
    pnuc = new G4NuclearStopping();
    pnuc->SetMaxKinEnergy(nielEnergyLimit);
  }

  // high energy limit for e+- scattering models
  G4double highEnergyLimit = G4EmParameters::Instance()->MscEnergyLimit();

  // Add gamma EM Processes
  G4ParticleDefinition* particle = G4Gamma::Gamma();

  // Photoelectric
  G4PhotoElectricEffect* pe = new G4PhotoElectricEffect();
  G4VEmModel* theLivermorePEModel = new G4LivermorePhotoElectricModel();
  pe->SetEmModel(theLivermorePEModel);

  // Compton scattering
  G4ComptonScattering* cs = new G4ComptonScattering;
  cs->SetEmModel(new G4KleinNishinaModel());
  G4VEmModel* theLowEPComptonModel = new G4LowEPComptonModel();
  theLowEPComptonModel->SetHighEnergyLimit(20 * CLHEP::MeV);
  cs->AddEmModel(0, theLowEPComptonModel);

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
  // e-/e+ msc gs with Mott-correction
  // (Mott-correction is set through G4EmParameters)
  G4GoudsmitSaundersonMscModel* msc1 = new G4GoudsmitSaundersonMscModel();
  G4WentzelVIModel* msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);
  msc->SetEmModel(msc1);
  msc->SetEmModel(msc2);

  G4eCoulombScatteringModel* ssm = new G4eCoulombScatteringModel();
  G4CoulombScattering* ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  // single scattering
  ssm = new G4eCoulombScatteringModel();
  ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  // ionisation
  G4eIonisation* eioni = new G4eIonisation();
  G4VEmModel* theIoniLiv = new G4LivermoreIonisationModel();
  theIoniLiv->SetHighEnergyLimit(0.1 * CLHEP::MeV);
  eioni->AddEmModel(0, theIoniLiv, new G4UniversalFluctuation());

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
  // e-/e+ msc gs with Mott-correction
  // (Mott-correction is set through G4EmParameters)
  msc1 = new G4GoudsmitSaundersonMscModel();
  msc2 = new G4WentzelVIModel();
  msc1->SetHighEnergyLimit(highEnergyLimit);
  msc2->SetLowEnergyLimit(highEnergyLimit);
  msc->SetEmModel(msc1);
  msc->SetEmModel(msc2);

  // single scattering
  ssm = new G4eCoulombScatteringModel();
  ss = new G4CoulombScattering();
  ss->SetEmModel(ssm);
  ss->SetMinKinEnergy(highEnergyLimit);
  ssm->SetLowEnergyLimit(highEnergyLimit);
  ssm->SetActivationLowEnergyLimit(highEnergyLimit);

  // ionisation
  eioni = new G4eIonisation();
  /*
  G4VEmModel* pen = new G4PenelopeIonisationModel();
  pen->SetHighEnergyLimit(0.1*CLHEP::MeV);
  eioni->AddEmModel(0, pen, new G4UniversalFluctuation());
  */
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
  ionIoni->SetEmModel(new G4IonParametrisedLossModel());
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);
  if (nullptr != pnuc)
    ph->RegisterProcess(pnuc, particle);

  // muons, hadrons, ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);
}

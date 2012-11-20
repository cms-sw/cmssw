#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysicsLPM.h"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4LossTableManager.hh"
#include "G4EmProcessOptions.hh"
#include "G4RegionStore.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4PairProductionRelModel.hh"

#include "G4hMultipleScattering.hh"
#include "G4eMultipleScattering.hh"
#include "G4MuMultipleScattering.hh"
#include "G4MscStepLimitType.hh"
#include "G4UrbanMscModel93.hh"
#include "G4WentzelVIModel.hh"
#include "G4CoulombScattering.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

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

CMSEmStandardPhysicsLPM::CMSEmStandardPhysicsLPM(const G4String& name, G4int ver):
  G4VPhysicsConstructor(name), verbose(ver) {
  G4LossTableManager::Instance();
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

void CMSEmStandardPhysicsLPM::ConstructProcess() 
{
  // Add standard EM Processes

  theParticleIterator->reset();
  while( (*theParticleIterator)() ){
    G4ParticleDefinition* particle = theParticleIterator->value();
    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();
    if(verbose > 1)
      G4cout << "### " << GetPhysicsName() << " instantiates for " 
	     << particleName << " at " << particle << G4endl;

    if (particleName == "gamma") {

      pmanager->AddDiscreteProcess(new G4PhotoElectricEffect);
      pmanager->AddDiscreteProcess(new G4ComptonScattering);
      G4GammaConversion* conv = new G4GammaConversion();
      G4PairProductionRelModel* mod = new G4PairProductionRelModel();
      mod->SetLowEnergyLimit(100*GeV);
      conv->AddEmModel(0, mod);
      pmanager->AddDiscreteProcess(conv);

    } else if (particleName == "e-") {

      G4eIonisation* eioni = new G4eIonisation();
      eioni->SetStepFunction(0.8, 1.0*mm);
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      msc->SetStepLimitType(fMinimal);
      msc->AddEmModel(0,new G4UrbanMscModel93());

      G4eBremsstrahlung* ebrem = new G4eBremsstrahlung();
      //ebrem->SetEmModel(new G4SeltzerBergerModel(), 1);
      // ebrem->SetEmModel(new G4eBremsstrahlungRelModel(), 2);
      //ebrem->EmModel(2)->SetLowEnergyLimit(GeV);

      pmanager->AddProcess(msc,                   -1, 1, 1);
      pmanager->AddProcess(eioni,                 -1, 2, 2);
      pmanager->AddProcess(ebrem,                 -1,-3, 3);

    } else if (particleName == "e+") {

      G4eIonisation* eioni = new G4eIonisation();
      eioni->SetStepFunction(0.8, 1.0*mm);
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      msc->SetStepLimitType(fMinimal);
      msc->AddEmModel(0,new G4UrbanMscModel93());

      G4eBremsstrahlung* ebrem = new G4eBremsstrahlung();
      //ebrem->SetEmModel(new G4SeltzerBergerModel95(), 1);
      //ebrem->SetEmModel(new G4eBremsstrahlungRelModel95(), 2);
      //ebrem->EmModel(2)->SetLowEnergyLimit(GeV);

      pmanager->AddProcess(msc,                     -1, 1, 1);
      pmanager->AddProcess(eioni,                   -1, 2, 2);
      pmanager->AddProcess(ebrem,                   -1,-3, 3);
      pmanager->AddProcess(new G4eplusAnnihilation,  0,-1, 4);

    } else if (particleName == "mu+" ||
               particleName == "mu-"    ) {

      G4MuMultipleScattering* mumsc = new G4MuMultipleScattering();
      mumsc->AddEmModel(0, new G4WentzelVIModel());
      pmanager->AddProcess(mumsc,                     -1, 1, 1);
      pmanager->AddProcess(new G4MuIonisation,        -1, 2, 2);
      pmanager->AddProcess(new G4MuBremsstrahlung,    -1,-3, 3);
      pmanager->AddProcess(new G4MuPairProduction,    -1,-4, 4);
      pmanager->AddProcess(new G4CoulombScattering,   -1,-4, 5);

    } else if (particleName == "alpha" ||
               particleName == "He3" ||
               particleName == "GenericIon") {

      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4ionIonisation,       -1, 2, 2);

    } else if (particleName == "pi+" ||
	       particleName == "kaon+" ||
	       particleName == "kaon-" ||
	       particleName == "proton" ||
	       particleName == "pi-" ) {

      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4hIonisation,         -1, 2, 2);
      pmanager->AddProcess(new G4hBremsstrahlung(),   -1,-3, 3);
      pmanager->AddProcess(new G4hPairProduction(),   -1,-4, 4);

    } else if (particleName == "B+" ||
	       particleName == "B-" ||
	       particleName == "D+" ||
	       particleName == "D-" ||
	       particleName == "Ds+" ||
	       particleName == "Ds-" ||
               particleName == "anti_lambda_c+" ||
               particleName == "anti_omega-" ||
               particleName == "anti_proton" ||
               particleName == "anti_sigma_c+" ||
               particleName == "anti_sigma_c++" ||
               particleName == "anti_sigma+" ||
               particleName == "anti_sigma-" ||
               particleName == "anti_xi_c+" ||
               particleName == "anti_xi-" ||
               particleName == "deuteron" ||
	       particleName == "lambda_c+" ||
               particleName == "omega-" ||
               particleName == "sigma_c+" ||
               particleName == "sigma_c++" ||
               particleName == "sigma+" ||
               particleName == "sigma-" ||
               particleName == "tau+" ||
               particleName == "tau-" ||
               particleName == "triton" ||
               particleName == "xi_c+" ||
               particleName == "xi-" ) {

      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4hIonisation,         -1, 2, 2);
    }
  }

  // Setup options
  //
  G4EmProcessOptions opt;
  opt.SetVerbose(verbose);

  // muon scattering
  opt.SetPolarAngleLimit(CLHEP::pi);

  // ApplyCuts
  //
  opt.SetApplyCuts(true);

  // Physics tables
  //
  opt.SetMinEnergy(100.*eV);
  opt.SetMaxEnergy(10.*TeV);
  opt.SetDEDXBinning(77);
  opt.SetLambdaBinning(77);

}

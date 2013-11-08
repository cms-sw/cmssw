#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics95.h"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4LossTableManager.hh"
#include "G4EmProcessOptions.hh"
#include "G4RegionStore.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4GammaConversion.hh"
#include "G4PairProductionRelModel.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4hMultipleScattering.hh"
#include "G4eMultipleScattering.hh"
#include "G4MscStepLimitType.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4MuMultipleScattering.hh"
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
#include "G4BuilderType.hh"

#include "G4SystemOfUnits.hh"

CMSEmStandardPhysics95::CMSEmStandardPhysics95(const G4String& name, G4int ver, std::string reg):
  G4VPhysicsConstructor(name), verbose(ver), region(reg) {
  G4LossTableManager::Instance();
  SetPhysicsType(bElectromagnetic);
}

CMSEmStandardPhysics95::~CMSEmStandardPhysics95() {}

void CMSEmStandardPhysics95::ConstructParticle() {
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

void CMSEmStandardPhysics95::ConstructProcess() 
{
  // Add standard EM Processes
  /*
  G4Region* reg = 0;
  if (region != " ") {
    G4RegionStore* regStore = G4RegionStore::GetInstance();
    reg = regStore->GetRegion(region, true);
  }
  */

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // Add standard EM Processes
  aParticleIterator->reset();
  while( (*aParticleIterator)() ){
    G4ParticleDefinition* particle = aParticleIterator->value();
    G4String particleName = particle->GetParticleName();
    if(verbose > 1)
      G4cout << "### " << GetPhysicsName() << " instantiates for " 
	     << particleName << G4endl;

    if (particleName == "gamma") {

      ph->RegisterProcess(new G4PhotoElectricEffect(), particle);
      ph->RegisterProcess(new G4ComptonScattering(), particle);
      G4GammaConversion* conv = new G4GammaConversion();
      G4PairProductionRelModel* mod = new G4PairProductionRelModel();
      mod->SetLowEnergyLimit(80*GeV);
      conv->AddEmModel(0, mod);
      ph->RegisterProcess(conv, particle);

    } else if (particleName == "e-") {

      G4eIonisation* eioni = new G4eIonisation();
      eioni->SetStepFunction(0.8, 1.0*mm);
      G4eMultipleScattering* msc = new G4eMultipleScattering();
      msc->SetStepLimitType(fMinimal);

      G4eBremsstrahlung* ebrem = new G4eBremsstrahlung();

      ph->RegisterProcess(msc, particle);
      ph->RegisterProcess(eioni, particle);
      ph->RegisterProcess(ebrem, particle);

    } else if (particleName == "e+") {

      G4eIonisation* eioni = new G4eIonisation();
      eioni->SetStepFunction(0.8, 1.0*mm);
      G4eMultipleScattering* msc = new G4eMultipleScattering();
      msc->SetStepLimitType(fMinimal);

      G4eBremsstrahlung* ebrem = new G4eBremsstrahlung();

      ph->RegisterProcess(msc, particle);
      ph->RegisterProcess(eioni, particle);
      ph->RegisterProcess(ebrem, particle);
      ph->RegisterProcess(new G4eplusAnnihilation(), particle);

    } else if (particleName == "mu+" ||
               particleName == "mu-"    ) {

      G4MuMultipleScattering* msc = new G4MuMultipleScattering();
      G4MuBremsstrahlung* mub = new G4MuBremsstrahlung();
      G4MuPairProduction* mup = new G4MuPairProduction();

      ph->RegisterProcess(msc, particle);
      ph->RegisterProcess(new G4MuIonisation(), particle);
      ph->RegisterProcess(mub, particle);
      ph->RegisterProcess(mup, particle);

    } else if (particleName == "alpha" || 
	       particleName == "He3" ) {

      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4ionIonisation(), particle);

    } else if (particleName == "GenericIon") {

      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4ionIonisation(), particle);

    } else if (particleName == "pi+" || 
	       particleName == "pi-" ) {

      G4hBremsstrahlung* pib = new G4hBremsstrahlung();
      G4hPairProduction* pip = new G4hPairProduction();

      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
      ph->RegisterProcess(pib, particle);
      ph->RegisterProcess(pip, particle);

    } else if (particleName == "kaon+" ||
               particleName == "kaon-" ) {

      G4hBremsstrahlung* kb = new G4hBremsstrahlung();
      G4hPairProduction* kp = new G4hPairProduction();

      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
      ph->RegisterProcess(kb, particle);
      ph->RegisterProcess(kp, particle);

    } else if (particleName == "proton" ) {

      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
      ph->RegisterProcess(new G4hBremsstrahlung(), particle);
      ph->RegisterProcess(new G4hPairProduction(), particle);

    } else if (particleName == "B+" ||
	       particleName == "B-" ||
	       particleName == "D+" ||
	       particleName == "D-" ||
	       particleName == "Ds+" ||
	       particleName == "Ds-" ||
               particleName == "anti_He3" ||
               particleName == "anti_alpha" ||
               particleName == "anti_deuteron" ||
               particleName == "anti_lambda_c+" ||
               particleName == "anti_omega-" ||
	       particleName == "anti_proton" ||
               particleName == "anti_sigma_c+" ||
               particleName == "anti_sigma_c++" ||
               particleName == "anti_sigma+" ||
               particleName == "anti_sigma-" ||
               particleName == "anti_triton" ||
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

      ph->RegisterProcess(new G4hMultipleScattering(), particle);
      ph->RegisterProcess(new G4hIonisation(), particle);
    }
  }

  // Setup options
  //
  G4EmProcessOptions opt;
  opt.SetVerbose(verbose);
  //  opt.SetPolarAngleLimit(CLHEP::pi);
  // ApplyCuts
  //
  opt.SetApplyCuts(true);
}

#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics95msc93.h"
#include "SimG4Core/PhysicsLists/interface/UrbanMscModel93.h"
#include "SimG4Core/PhysicsLists/interface/EmParticleList.h"
#include "G4EmParameters.hh"
#include "G4ParticleTable.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4LossTableManager.hh"
#include "G4RegionStore.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4PairProductionRelModel.hh"

#include "G4hMultipleScattering.hh"
#include "G4eMultipleScattering.hh"
#include "G4MscStepLimitType.hh"

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

#include "G4BuilderType.hh"
#include "G4SystemOfUnits.hh"

CMSEmStandardPhysics95msc93::CMSEmStandardPhysics95msc93(const G4String& name, 
							 G4int ver, 
							 const std::string& reg)
: G4VPhysicsConstructor(name), verbose(ver), region(reg) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(verbose);
  param->SetApplyCuts(true);
  param->SetMscRangeFactor(0.2);
  param->SetMscStepLimitType(fMinimal);
  SetPhysicsType(bElectromagnetic);
  G4LossTableManager::Instance();
}

CMSEmStandardPhysics95msc93::~CMSEmStandardPhysics95msc93() {}

void CMSEmStandardPhysics95msc93::ConstructParticle() {
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

void CMSEmStandardPhysics95msc93::ConstructProcess() 
{
  // Add standard EM Processes

  // muon & hadron bremsstrahlung and pair production
  G4MuBremsstrahlung* mub = nullptr;
  G4MuPairProduction* mup = nullptr;
  G4hBremsstrahlung* pib = nullptr;
  G4hPairProduction* pip = nullptr;
  G4hBremsstrahlung* kb = nullptr;
  G4hPairProduction* kp = nullptr;
  G4hBremsstrahlung* pb = nullptr;
  G4hPairProduction* pp = nullptr;

  G4hMultipleScattering* hmsc = nullptr;

  // This EM builder takes default models of Geant4 10 EMV.
  // Multiple scattering by Urban for all particles
  // except e+e- for which the Urban93 model is used

  G4ParticleTable* table = G4ParticleTable::GetParticleTable();
  EmParticleList emList;
  for(const auto& particleName : emList.PartNames()) {
    G4ParticleDefinition* particle = table->FindParticle(particleName);
    G4ProcessManager* pmanager = particle->GetProcessManager();
    if(verbose > 1)
      G4cout << "### " << GetPhysicsName() << " instantiates for " 
	     << particleName << " at " << particle << G4endl;

    if (particleName == "gamma") {

      pmanager->AddDiscreteProcess(new G4PhotoElectricEffect);
      pmanager->AddDiscreteProcess(new G4ComptonScattering);
      pmanager->AddDiscreteProcess(new G4GammaConversion());

    } else if (particleName == "e-") {

      G4eIonisation* eioni = new G4eIonisation();
      eioni->SetStepFunction(0.8, 1.0*mm);
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      msc->SetStepLimitType(fMinimal);
      msc->AddEmModel(0,new UrbanMscModel93());

      G4eBremsstrahlung* ebrem = new G4eBremsstrahlung();

      pmanager->AddProcess(msc,                   -1, 1, 1);
      pmanager->AddProcess(eioni,                 -1, 2, 2);
      pmanager->AddProcess(ebrem,                 -1,-3, 3);

    } else if (particleName == "e+") {

      G4eIonisation* eioni = new G4eIonisation();
      eioni->SetStepFunction(0.8, 1.0*mm);

      G4eMultipleScattering* msc = new G4eMultipleScattering;
      msc->SetStepLimitType(fMinimal);
      msc->AddEmModel(0,new UrbanMscModel93());

      G4eBremsstrahlung* ebrem = new G4eBremsstrahlung();

      pmanager->AddProcess(msc,                     -1, 1, 1);
      pmanager->AddProcess(eioni,                   -1, 2, 2);
      pmanager->AddProcess(ebrem,                   -1,-3, 3);
      pmanager->AddProcess(new G4eplusAnnihilation,  0,-1, 4);

    } else if (particleName == "mu+" ||
               particleName == "mu-"    ) {

      if(nullptr == mub) {
	mub = new G4MuBremsstrahlung();
	mup = new G4MuPairProduction();
      }
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4MuIonisation,        -1, 2, 2);
      pmanager->AddProcess(mub,    -1,-3, 3);
      pmanager->AddProcess(mup,    -1,-4, 4);

    } else if (particleName == "alpha" ||
               particleName == "He3" ||
               particleName == "GenericIon") {

      if(nullptr == hmsc) {
	hmsc = new G4hMultipleScattering("ionmsc");
      }
      pmanager->AddProcess(hmsc,                  -1, 1, 1);
      pmanager->AddProcess(new G4ionIonisation,   -1, 2, 2);

    } else if (particleName == "pi+" ||
	       particleName == "pi-" ) {

      if(nullptr == pib) {
	pib = new G4hBremsstrahlung();
	pip = new G4hPairProduction();
      }
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4hIonisation,         -1, 2, 2);
      pmanager->AddProcess(pib,   -1,-3, 3);
      pmanager->AddProcess(pip,   -1,-4, 4);

    } else if (particleName == "kaon+" ||
	       particleName == "kaon-"  ) {

      if(nullptr == kb) {
	kb = new G4hBremsstrahlung();
	kp = new G4hPairProduction();
      }
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4hIonisation,         -1, 2, 2);
      pmanager->AddProcess(kb,   -1,-3, 3);
      pmanager->AddProcess(kp,   -1,-4, 4);

    } else if (particleName == "proton" || 
	       particleName == "anti_proton" ) {
      if(nullptr == pb) {
	pb = new G4hBremsstrahlung();
	pp = new G4hPairProduction();
      }
      pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
      pmanager->AddProcess(new G4hIonisation,         -1, 2, 2);
      pmanager->AddProcess(pb,   -1,-3, 3);
      pmanager->AddProcess(pp,   -1,-4, 4);

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

      if(nullptr == hmsc) {
	hmsc = new G4hMultipleScattering("ionmsc");
      }
      pmanager->AddProcess(hmsc,                -1, 1, 1);
      pmanager->AddProcess(new G4hIonisation,   -1, 2, 2);
    }
  }
}

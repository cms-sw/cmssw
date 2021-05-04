//
#include "SimG4Core/PhysicsLists/interface/CMSHyperonFTFPBuilder.h"

#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4CrossSectionInelastic.hh"
#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4HadronicParameters.hh"

#include "G4LambdaInelasticProcess.hh"
#include "G4AntiLambdaInelasticProcess.hh"
#include "G4SigmaPlusInelasticProcess.hh"
#include "G4SigmaMinusInelasticProcess.hh"
#include "G4AntiSigmaPlusInelasticProcess.hh"
#include "G4AntiSigmaMinusInelasticProcess.hh"
#include "G4XiZeroInelasticProcess.hh"
#include "G4XiMinusInelasticProcess.hh"
#include "G4AntiXiZeroInelasticProcess.hh"
#include "G4AntiXiMinusInelasticProcess.hh"
#include "G4OmegaMinusInelasticProcess.hh"
#include "G4AntiOmegaMinusInelasticProcess.hh"

#include "G4TheoFSGenerator.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4FTFModel.hh"
#include "G4LundStringFragmentation.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4CascadeInterface.hh"

CMSHyperonFTFPBuilder::CMSHyperonFTFPBuilder() {}

CMSHyperonFTFPBuilder::~CMSHyperonFTFPBuilder() {}

void CMSHyperonFTFPBuilder::Build() {
  // Hyperon : Bertini at low energies, then FTFP

  auto HyperonFTFP = new G4TheoFSGenerator("FTFP");

  HyperonFTFP->SetMinEnergy(G4HadronicParameters::Instance()->GetMinEnergyTransitionFTF_Cascade());
  HyperonFTFP->SetMaxEnergy(G4HadronicParameters::Instance()->GetMaxEnergy());

  auto theStringModel = new G4FTFModel;
  auto theStringDecay = new G4ExcitedStringDecay(new G4LundStringFragmentation());
  theStringModel->SetFragmentationModel(theStringDecay);

  auto theCascade = new G4GeneratorPrecompoundInterface;

  HyperonFTFP->SetTransport(theCascade);
  HyperonFTFP->SetHighEnergyGenerator(theStringModel);

  auto theBertini = new G4CascadeInterface;
  theBertini->SetMinEnergy(0.0);
  theBertini->SetMaxEnergy(G4HadronicParameters::Instance()->GetMaxEnergyTransitionFTF_Cascade());

  // AntiHyperons: Use FTFP for full energy range, starting at 0.

  auto AntiHyperonFTFP = new G4TheoFSGenerator("FTFP");
  AntiHyperonFTFP->SetMinEnergy(0.0);
  AntiHyperonFTFP->SetMaxEnergy(G4HadronicParameters::Instance()->GetMaxEnergy());
  AntiHyperonFTFP->SetTransport(theCascade);
  AntiHyperonFTFP->SetHighEnergyGenerator(theStringModel);

  // use Glauber-Gribov cross sections
  auto theInelasticCrossSection = new G4CrossSectionInelastic(new G4ComponentGGHadronNucleusXsc);

  G4ProcessManager* aProcMan = nullptr;

  // Lambda
  auto theLambdaInelastic = new G4LambdaInelasticProcess();
  theLambdaInelastic->RegisterMe(theBertini);
  theLambdaInelastic->RegisterMe(HyperonFTFP);
  theLambdaInelastic->AddDataSet(theInelasticCrossSection);
  aProcMan = G4Lambda::Lambda()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theLambdaInelastic);

  // AntiLambda
  auto theAntiLambdaInelastic = new G4AntiLambdaInelasticProcess();
  theAntiLambdaInelastic->RegisterMe(AntiHyperonFTFP);
  theAntiLambdaInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4AntiLambda::AntiLambda()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theAntiLambdaInelastic);

  // SigmaMinus
  auto theSigmaMinusInelastic = new G4SigmaMinusInelasticProcess();
  theSigmaMinusInelastic->RegisterMe(theBertini);
  theSigmaMinusInelastic->RegisterMe(HyperonFTFP);
  theSigmaMinusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4SigmaMinus::SigmaMinus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theSigmaMinusInelastic);

  // anti-SigmaMinus
  auto theAntiSigmaMinusInelastic = new G4AntiSigmaMinusInelasticProcess();
  theAntiSigmaMinusInelastic->RegisterMe(AntiHyperonFTFP);
  theAntiSigmaMinusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4AntiSigmaMinus::AntiSigmaMinus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theAntiSigmaMinusInelastic);

  // SigmaPlus
  auto theSigmaPlusInelastic = new G4SigmaPlusInelasticProcess();
  theSigmaPlusInelastic->RegisterMe(theBertini);
  theSigmaPlusInelastic->RegisterMe(HyperonFTFP);
  theSigmaPlusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4SigmaPlus::SigmaPlus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theSigmaPlusInelastic);

  // anti-SigmaPlus
  auto theAntiSigmaPlusInelastic = new G4AntiSigmaPlusInelasticProcess();
  theAntiSigmaPlusInelastic->RegisterMe(AntiHyperonFTFP);
  theAntiSigmaPlusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4AntiSigmaPlus::AntiSigmaPlus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theAntiSigmaPlusInelastic);

  // XiMinus
  auto theXiMinusInelastic = new G4XiMinusInelasticProcess();
  theXiMinusInelastic->RegisterMe(theBertini);
  theXiMinusInelastic->RegisterMe(HyperonFTFP);
  theXiMinusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4XiMinus::XiMinus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theXiMinusInelastic);

  // anti-XiMinus
  auto theAntiXiMinusInelastic = new G4AntiXiMinusInelasticProcess();
  theAntiXiMinusInelastic->RegisterMe(AntiHyperonFTFP);
  theAntiXiMinusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4AntiXiMinus::AntiXiMinus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theAntiXiMinusInelastic);

  // XiZero
  auto theXiZeroInelastic = new G4XiZeroInelasticProcess();
  theXiZeroInelastic->RegisterMe(theBertini);
  theXiZeroInelastic->RegisterMe(HyperonFTFP);
  theXiZeroInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4XiZero::XiZero()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theXiZeroInelastic);

  // anti-XiZero
  auto theAntiXiZeroInelastic = new G4AntiXiZeroInelasticProcess();
  theAntiXiZeroInelastic->RegisterMe(AntiHyperonFTFP);
  theAntiXiZeroInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4AntiXiZero::AntiXiZero()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theAntiXiZeroInelastic);

  // OmegaMinus
  auto theOmegaMinusInelastic = new G4OmegaMinusInelasticProcess();
  theOmegaMinusInelastic->RegisterMe(theBertini);
  theOmegaMinusInelastic->RegisterMe(HyperonFTFP);
  theOmegaMinusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4OmegaMinus::OmegaMinus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theOmegaMinusInelastic);

  // anti-OmegaMinus
  auto theAntiOmegaMinusInelastic = new G4AntiOmegaMinusInelasticProcess();
  theAntiOmegaMinusInelastic->RegisterMe(AntiHyperonFTFP);
  theAntiOmegaMinusInelastic->AddDataSet(theInelasticCrossSection);

  aProcMan = G4AntiOmegaMinus::AntiOmegaMinus()->GetProcessManager();
  aProcMan->AddDiscreteProcess(theAntiOmegaMinusInelastic);
}

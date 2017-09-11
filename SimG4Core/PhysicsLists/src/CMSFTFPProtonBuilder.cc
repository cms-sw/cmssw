#include "SimG4Core/PhysicsLists/interface/CMSFTFPProtonBuilder.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ProtonInelasticCrossSection.hh"
#include "G4SystemOfUnits.hh"

CMSFTFPProtonBuilder::CMSFTFPProtonBuilder(G4bool quasiElastic) 
{
  theMin = 4*GeV;
  theMax = 100.*TeV; 
  theModel = new G4TheoFSGenerator("FTFP");

  theStringModel = new G4FTFModel;
  theStringDecay = new G4ExcitedStringDecay(new G4LundStringFragmentation);
  theStringModel->SetFragmentationModel(theStringDecay);

  theCascade = new G4GeneratorPrecompoundInterface;
  thePreEquilib = new G4PreCompoundModel(new G4ExcitationHandler);
  theCascade->SetDeExcitation(thePreEquilib);  

  theModel->SetHighEnergyGenerator(theStringModel);
  if (quasiElastic)
  {
     theQuasiElastic=new G4QuasiElasticChannel;
     theModel->SetQuasiElasticChannel(theQuasiElastic);
  } else 
  {  theQuasiElastic=nullptr;}  

  theModel->SetTransport(theCascade);
  theModel->SetMinEnergy(theMin);
  theModel->SetMaxEnergy(100*TeV);
}

void CMSFTFPProtonBuilder::Build(G4ProtonInelasticProcess * aP)
{
  theModel->SetMinEnergy(theMin);
  theModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theModel);
  aP->AddDataSet(new G4ProtonInelasticCrossSection);  
}

CMSFTFPProtonBuilder::
~CMSFTFPProtonBuilder() 
{
  delete theStringDecay;
  delete theStringModel;
  delete theModel;
  delete theCascade;
  if ( theQuasiElastic ) delete theQuasiElastic;
}

void CMSFTFPProtonBuilder::
Build(G4HadronElasticProcess * )
{
}

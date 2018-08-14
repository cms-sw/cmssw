#include "SimG4Core/PhysicsLists/interface/CMSFTFPNeutronBuilder.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4NeutronInelasticCrossSection.hh"
#include "G4SystemOfUnits.hh"

CMSFTFPNeutronBuilder::CMSFTFPNeutronBuilder(G4bool quasiElastic) 
{
  theMin =   4*GeV;
  theMax = 100*TeV;
  theModel = new G4TheoFSGenerator("FTFP");

  theStringModel = new G4FTFModel;
  theStringDecay = new G4ExcitedStringDecay(new G4LundStringFragmentation);
  theStringModel->SetFragmentationModel(theStringDecay);

  theCascade = new G4GeneratorPrecompoundInterface;
  thePreEquilib = new G4PreCompoundModel(new G4ExcitationHandler);
  theCascade->SetDeExcitation(thePreEquilib);  

  theModel->SetTransport(theCascade);

  theModel->SetHighEnergyGenerator(theStringModel);
  if (quasiElastic)
  {
     theQuasiElastic=new G4QuasiElasticChannel;
     theModel->SetQuasiElasticChannel(theQuasiElastic);
  } else 
  {  theQuasiElastic=nullptr;}  

  theModel->SetMinEnergy(theMin);
  theModel->SetMaxEnergy(100*TeV);
}

CMSFTFPNeutronBuilder::
~CMSFTFPNeutronBuilder() 
{
  delete theStringDecay;
  delete theStringModel;
  delete thePreEquilib;
  delete theCascade;
  if ( theQuasiElastic ) delete theQuasiElastic;
}

void CMSFTFPNeutronBuilder::
Build(G4HadronElasticProcess * )
{
}

void CMSFTFPNeutronBuilder::
Build(G4HadronFissionProcess * )
{
}

void CMSFTFPNeutronBuilder::
Build(G4HadronCaptureProcess * )
{
}

void CMSFTFPNeutronBuilder::
Build(G4NeutronInelasticProcess * aP)
{
  theModel->SetMinEnergy(theMin);
  theModel->SetMaxEnergy(theMax);
  aP->RegisterMe(theModel);
  aP->AddDataSet(new G4NeutronInelasticCrossSection);  
}

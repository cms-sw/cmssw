
#include "G4QGSPSIMPBuilder.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4ExcitationHandler.hh"

#include "G4SIMPInelasticXS.hh"
#include "G4SIMP.hh"


G4QGSPSIMPBuilder::
G4QGSPSIMPBuilder(G4bool quasiElastic) 
{
  theMin = 12*GeV;

  theModel = new G4TheoFSGenerator("QGSP");

  theStringModel = new G4QGSModel< G4QGSParticipants >;
  theStringDecay = new G4ExcitedStringDecay(theQGSM = new G4QGSMFragmentation);
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
  {  theQuasiElastic=0;}  
}

G4QGSPSIMPBuilder::~G4QGSPSIMPBuilder() 
{
  delete theStringDecay;
  delete theStringModel;
  delete thePreEquilib;
  delete theCascade;
  if ( theQuasiElastic ) delete theQuasiElastic;
  delete theModel;
  delete theQGSM;
}

void G4QGSPSIMPBuilder::
Build(G4HadronElasticProcess * )
{
}

void G4QGSPSIMPBuilder::
Build(G4HadronFissionProcess * )
{
}

void G4QGSPSIMPBuilder::
Build(G4HadronCaptureProcess * )
{
}

void G4QGSPSIMPBuilder::
Build(G4SIMPInelasticProcess * aP)
{
  theModel->SetMinEnergy(theMin);
  theModel->SetMaxEnergy(100*TeV);
  aP->RegisterMe(theModel);
  aP->AddDataSet(new G4SIMPInelasticXS());
}

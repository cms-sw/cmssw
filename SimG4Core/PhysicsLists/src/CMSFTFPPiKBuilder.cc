#include "SimG4Core/PhysicsLists/interface/CMSFTFPPiKBuilder.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4SystemOfUnits.hh"

CMSFTFPPiKBuilder::CMSFTFPPiKBuilder(G4bool quasiElastic) 
 {
   thePiData = new G4PiNuclearCrossSection;
   theMin = 4*GeV;
   theMax = 100*TeV;
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

CMSFTFPPiKBuilder::~CMSFTFPPiKBuilder() 
 {
   delete theCascade;
   delete theStringDecay;
   delete theStringModel;
   delete theModel;
   if ( theQuasiElastic ) delete theQuasiElastic;
 }

void CMSFTFPPiKBuilder::
Build(G4HadronElasticProcess * ) {}

void CMSFTFPPiKBuilder::
Build(G4PionPlusInelasticProcess * aP)
 {
   theModel->SetMinEnergy(theMin);
   theModel->SetMaxEnergy(theMax);
   aP->AddDataSet(thePiData);
   aP->RegisterMe(theModel);
 }

void CMSFTFPPiKBuilder::
Build(G4PionMinusInelasticProcess * aP)
 {
   theModel->SetMinEnergy(theMin);
   theModel->SetMaxEnergy(theMax);
   aP->AddDataSet(thePiData);
   aP->RegisterMe(theModel);
 }

void CMSFTFPPiKBuilder::
Build(G4KaonPlusInelasticProcess * aP)
 {
   theModel->SetMinEnergy(theMin);
   theModel->SetMaxEnergy(theMax);
   aP->RegisterMe(theModel);
 }

void CMSFTFPPiKBuilder::
Build(G4KaonMinusInelasticProcess * aP)
 {
   theModel->SetMinEnergy(theMin);
   theModel->SetMaxEnergy(theMax);
   aP->RegisterMe(theModel);
 }

void CMSFTFPPiKBuilder::
Build(G4KaonZeroLInelasticProcess * aP)
 {
   theModel->SetMinEnergy(theMin);
   theModel->SetMaxEnergy(theMax);
   aP->RegisterMe(theModel);
 }

void CMSFTFPPiKBuilder::
Build(G4KaonZeroSInelasticProcess * aP)
 {
   theModel->SetMinEnergy(theMin);
   theModel->SetMaxEnergy(theMax);
   aP->RegisterMe(theModel);
 }

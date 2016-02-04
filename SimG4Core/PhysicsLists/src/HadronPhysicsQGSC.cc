#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSC.hh"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>   
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

HadronPhysicsQGSC::HadronPhysicsQGSC(G4int)
                    :  G4VPhysicsConstructor("hInelastic QGSC")
		     , QuasiElastic(true)
{}

HadronPhysicsQGSC::HadronPhysicsQGSC(const G4String& name, G4bool quasiElastic)
                    :  G4VPhysicsConstructor(name)  , QuasiElastic(quasiElastic)
{}

void HadronPhysicsQGSC::CreateModels()
{
  theNeutrons=new G4NeutronBuilder;
  theNeutrons->RegisterMe(theQGSCNeutron=new G4QGSCNeutronBuilder(QuasiElastic));
  theNeutrons->RegisterMe(theLEPNeutron=new G4LEPNeutronBuilder);
  theLEPNeutron->SetMaxInelasticEnergy(25*GeV);

  thePro=new G4ProtonBuilder;
  thePro->RegisterMe(theQGSCPro=new G4QGSCProtonBuilder(QuasiElastic));
  thePro->RegisterMe(theLEPPro=new G4LEPProtonBuilder);
  theLEPPro->SetMaxEnergy(25*GeV);

  thePiK=new G4PiKBuilder;
  thePiK->RegisterMe(theQGSCPiK=new G4QGSCPiKBuilder(QuasiElastic));
  thePiK->RegisterMe(theLEPPiK=new G4LEPPiKBuilder);
  theLEPPiK->SetMaxEnergy(25*GeV);
  
  theMiscLHEP=new G4MiscLHEPBuilder;
}

HadronPhysicsQGSC::~HadronPhysicsQGSC() 
{
   delete theMiscLHEP;
   delete theQGSCPro;
   delete theLEPPro;
   delete thePro;
   delete theQGSCNeutron;
   delete theNeutrons;
   delete theQGSCPiK;
   delete theLEPPiK;
   delete thePiK;
}

void HadronPhysicsQGSC::ConstructParticle()
{
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"
void HadronPhysicsQGSC::ConstructProcess()
{
  CreateModels();
  theNeutrons->Build();
  thePro->Build();
  thePiK->Build();
  theMiscLHEP->Build();
}


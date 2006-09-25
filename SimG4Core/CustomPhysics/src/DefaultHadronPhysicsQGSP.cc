#include "DefaultHadronPhysicsQGSP.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

HadronPhysicsQGSP::HadronPhysicsQGSP(const G4String& name)
                    :  G4VPhysicsConstructor(name) 
{
#ifdef G4v7
  theNeutrons.RegisterMe(&theQGSPNeutron);
  theNeutrons.RegisterMe(&theLEPNeutron);
  theLEPNeutron.SetMaxInelasticEnergy(25*GeV);

  thePro.RegisterMe(&theQGSPPro);
  thePro.RegisterMe(&theLEPPro);
  theLEPPro.SetMaxEnergy(25*GeV);
  
  thePiK.RegisterMe(&theQGSPPiK);
  thePiK.RegisterMe(&theLEPPiK);
  theLEPPiK.SetMaxEnergy(25*GeV);
#endif
}

HadronPhysicsQGSP::~HadronPhysicsQGSP() {
#ifndef G4v7
  delete theStoppingHadron;
  delete theMiscLHEP;
  delete theQGSPNeutron;
  delete theLEPNeutron;
  delete theQGSPPro;
  delete theLEPPro;
  delete thePro;
  delete theQGSPPiK;
  delete theLEPPiK;
  delete thePiK;
#endif
}
#ifndef G4v7

void HadronPhysicsQGSP::CreateModels() {
  theNeutrons=new G4NeutronBuilder;
  theNeutrons->RegisterMe(theQGSPNeutron=new G4QGSPNeutronBuilder);
  theNeutrons->RegisterMe(theLEPNeutron=new G4LEPNeutronBuilder);
  theLEPNeutron->SetMaxInelasticEnergy(25*GeV);  

  thePro=new G4ProtonBuilder;
  thePro->RegisterMe(theQGSPPro=new G4QGSPProtonBuilder);
  thePro->RegisterMe(theLEPPro=new G4LEPProtonBuilder);
  theLEPPro->SetMaxEnergy(25*GeV);
  
  thePiK=new G4PiKBuilder;
  thePiK->RegisterMe(theQGSPPiK=new G4QGSPPiKBuilder);
  thePiK->RegisterMe(theLEPPiK=new G4LEPPiKBuilder);
  theLEPPiK->SetMaxEnergy(25*GeV);
  
  theMiscLHEP=new G4MiscLHEPBuilder;
  theStoppingHadron=new G4StoppingHadronBuilder;
}
#endif

void HadronPhysicsQGSP::ConstructParticle()
{
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"

void HadronPhysicsQGSP::ConstructProcess()
{
#ifdef G4v7
  theNeutrons.Build();
  thePro.Build();
  thePiK.Build();
  theMiscLHEP.Build();
  theStoppingHadron.Build();
  theHadronQED.Build();
#else
  CreateModels();
  theNeutrons->Build();
  thePro->Build();
  thePiK->Build();
  theMiscLHEP->Build();
  theStoppingHadron->Build();
#endif
}
// 2002 by J.P. Wellisch

#include "DefaultHadronPhysicsQGSP.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

HadronPhysicsQGSP::HadronPhysicsQGSP(const G4String& name)
:  G4VPhysicsConstructor(name) 
{
	theNeutrons.RegisterMe(&theQGSPNeutron);
	theNeutrons.RegisterMe(&theLEPNeutron);
	theLEPNeutron.SetMaxInelasticEnergy(25*GeV);
	
	thePro.RegisterMe(&theQGSPPro);
	thePro.RegisterMe(&theLEPPro);
	theLEPPro.SetMaxEnergy(25*GeV);
	
	thePiK.RegisterMe(&theQGSPPiK);
	thePiK.RegisterMe(&theLEPPiK);
	theLEPPiK.SetMaxEnergy(25*GeV);
}

HadronPhysicsQGSP::~HadronPhysicsQGSP() {}

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
	theNeutrons.Build();
	thePro.Build();
	thePiK.Build();
	theMiscLHEP.Build();
	theStoppingHadron.Build();
	theHadronQED.Build();
}

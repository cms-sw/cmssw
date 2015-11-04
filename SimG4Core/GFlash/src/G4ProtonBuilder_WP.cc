//
//---------------------------------------------------------------------------
//
// ClassName:   G4PiKBuilder
//
// Author: 2002 J.P. Wellisch
//
// Modified:
// 16.11.2005 G.Folger: don't  keep processes as data members, but new these
// 13.06.2006 G.Folger: (re)move elastic scatterring 
//
//----------------------------------------------------------------------------
//
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"

#include "SimG4Core/GFlash/interface/GflashHadronWrapperProcess.h"
#include "SimG4Core/GFlash/interface/G4ProtonBuilder_WP.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
 void G4ProtonBuilder_WP::Build()
 {
   wasActivated = true;
   std::vector<G4VProtonBuilder *>::iterator i;
   for(i=theModelCollections.begin(); i!=theModelCollections.end(); i++)
   {
     (*i)->Build(theProtonInelastic);
   }
   G4ProcessManager * theProcMan = G4Proton::Proton()->GetProcessManager();
   //   theProcMan->AddDiscreteProcess(theProtonInelastic);
   edm::LogInfo("SimG4CoreGFlash") << " Adding GflashHadronWrapperProcess (G4wrapperProcess) for G4ProtonInelasticProcess";
   theWrappedProtonInelastic->RegisterProcess(theProtonInelastic);
   theProcMan->AddDiscreteProcess(theWrappedProtonInelastic);
 
}

 G4ProtonBuilder_WP::
 G4ProtonBuilder_WP(): wasActivated(false)  
 {
   theProtonInelastic=new G4ProtonInelasticProcess;

   theWrappedProtonInelastic=new GflashHadronWrapperProcess("G4ProtonInelasticProcess");
 }

 G4ProtonBuilder_WP::
 ~G4ProtonBuilder_WP() 
 {
   delete theProtonInelastic;
 }

 // 2002 by J.P. Wellisch
 // 2009 Modified for CMS GflashHadronWrapperProcess

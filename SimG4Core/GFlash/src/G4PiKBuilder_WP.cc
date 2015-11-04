//
//---------------------------------------------------------------------------
//
// ClassName:   G4PiKBuilder_WP
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
#include "SimG4Core/GFlash/interface/G4PiKBuilder_WP.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

G4PiKBuilder_WP::
G4PiKBuilder_WP(): wasActivated(false) 
{  
  thePionPlusInelastic=new G4PionPlusInelasticProcess;
  thePionMinusInelastic=new G4PionMinusInelasticProcess;
  theKaonPlusInelastic=new G4KaonPlusInelasticProcess;
  theKaonMinusInelastic=new G4KaonMinusInelasticProcess;
  theKaonZeroLInelastic=new G4KaonZeroLInelasticProcess;
  theKaonZeroSInelastic=new G4KaonZeroSInelasticProcess;

  theWrappedPionPlusInelastic=new GflashHadronWrapperProcess("G4PionPlusInelasticProcess");
  theWrappedPionMinusInelastic=new GflashHadronWrapperProcess("G4PionMinusInelasticProcess");
  theWrappedKaonPlusInelastic=new GflashHadronWrapperProcess("G4KaonPlusInelasticProcess");
  theWrappedKaonMinusInelastic=new GflashHadronWrapperProcess("G4KaonMinusInelasticProcess");
}

G4PiKBuilder_WP::
~G4PiKBuilder_WP(){
  delete thePionPlusInelastic;
  delete thePionMinusInelastic;
  delete theKaonPlusInelastic;
  delete theKaonMinusInelastic;
  delete theKaonZeroLInelastic;
  delete theKaonZeroSInelastic;

  //  delete theWrappedPionPlusInelastic;
  //  delete theWrappedPionMinusInelastic;
}

void G4PiKBuilder_WP::
Build()
{
  wasActivated = true;

  std::vector<G4VPiKBuilder *>::iterator i;
  for(i=theModelCollections.begin(); i!=theModelCollections.end(); i++)
  {
    (*i)->Build(thePionPlusInelastic);
    (*i)->Build(thePionMinusInelastic);
    (*i)->Build(theKaonPlusInelastic);
    (*i)->Build(theKaonMinusInelastic);
    (*i)->Build(theKaonZeroLInelastic);
    (*i)->Build(theKaonZeroSInelastic);
  }
  G4ProcessManager * theProcMan;
  theProcMan = G4PionPlus::PionPlus()->GetProcessManager();
  //  theProcMan->AddDiscreteProcess(thePionPlusInelastic);
  edm::LogInfo("SimG4CoreGFlash") << " Adding GflashHadronWrapperProcess (G4wrapperProcess) for G4PionPlusInelasticProcess";
  theWrappedPionPlusInelastic->RegisterProcess(thePionPlusInelastic);
  theProcMan->AddDiscreteProcess(theWrappedPionPlusInelastic);
  
  theProcMan = G4PionMinus::PionMinus()->GetProcessManager();
  //  theProcMan->AddDiscreteProcess(thePionMinusInelastic);
  edm::LogInfo("SimG4CoreGFlash") << " Adding GflashHadronWrapperProcess (G4wrapperProcess) for G4PionMinusInelasticProcess";
  theWrappedPionMinusInelastic->RegisterProcess(thePionMinusInelastic);
  theProcMan->AddDiscreteProcess(theWrappedPionMinusInelastic);
  
  theProcMan = G4KaonPlus::KaonPlus()->GetProcessManager();
  //  theProcMan->AddDiscreteProcess(theKaonPlusInelastic);
  edm::LogInfo("SimG4CoreGFlash") << " Adding GflashHadronWrapperProcess (G4wrapperProcess) for G4KaonPlusInelasticProcess";
  theWrappedKaonPlusInelastic->RegisterProcess(theKaonPlusInelastic);
  theProcMan->AddDiscreteProcess(theWrappedKaonPlusInelastic);
  
  theProcMan = G4KaonMinus::KaonMinus()->GetProcessManager();
  //  theProcMan->AddDiscreteProcess(theKaonMinusInelastic);
  edm::LogInfo("SimG4CoreGFlash") << " Adding GflashHadronWrapperProcess (G4wrapperProcess) for G4KaonMinusInelasticProcess";
  theWrappedKaonMinusInelastic->RegisterProcess(theKaonMinusInelastic);
  theProcMan->AddDiscreteProcess(theWrappedKaonMinusInelastic);
  
  theProcMan = G4KaonZeroLong::KaonZeroLong()->GetProcessManager();
  theProcMan->AddDiscreteProcess(theKaonZeroLInelastic);
  
  theProcMan = G4KaonZeroShort::KaonZeroShort()->GetProcessManager();
  theProcMan->AddDiscreteProcess(theKaonZeroSInelastic);
}
// 2002 by J.P. Wellisch
// 2008 Modified for GflashHadronWrapperProcess

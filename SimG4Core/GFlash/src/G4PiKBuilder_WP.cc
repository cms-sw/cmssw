//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: G4PiKBuilder_WP.cc,v 1.5 2013/05/30 21:10:49 gartung Exp $
// GEANT4 tag $Name: CMSSW_6_2_0 $
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

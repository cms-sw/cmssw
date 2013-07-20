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
// $Id: G4ProtonBuilder_WP.cc,v 1.3 2013/05/30 21:10:49 gartung Exp $
// GEANT4 tag $Name: CMSSW_6_2_0 $
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

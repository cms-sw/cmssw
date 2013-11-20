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
//
// GEANT4 tag $Name: V04-05-01 $
//
//---------------------------------------------------------------------------
//
// ClassName:   HadronPhysicsQGSP_WP
//
// Author: 2002 J.P. Wellisch
//
// Modified:
// 21.11.2005 G.Folger:  migration to non static particles
// 08.06.2006 V.Ivanchenko: remove stopping
// 30.03.2007 G.Folger: Add code for quasielastic
//

#include "SimG4Core/GFlash/interface/HadronPhysicsQGSP_WP.h"
#include "SimG4Core/GFlash/interface/G4PiKBuilder_WP.h"
#include "SimG4Core/GFlash/interface/G4ProtonBuilder_WP.h"
#include "SimG4Core/GFlash/interface/G4MiscLHEPBuilder_WP.h"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>   
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4SystemOfUnits.hh"

HadronPhysicsQGSP_WP::HadronPhysicsQGSP_WP(const G4String& name, G4bool quasiElastic)
                 :  G4VPhysicsConstructor(name) , QuasiElastic(quasiElastic)
{}

void HadronPhysicsQGSP_WP::CreateModels()
{
  theNeutrons=new G4NeutronBuilder;
  theQGSPNeutron=new G4QGSPNeutronBuilder(QuasiElastic);
  theNeutrons->RegisterMe(theQGSPNeutron);
  // theNeutrons->RegisterMe(theLEPNeutron=new G4LEPNeutronBuilder);
  // theLEPNeutron->SetMaxInelasticEnergy(25*GeV);  

  thePro=new G4ProtonBuilder_WP;
  theQGSPPro=new G4QGSPProtonBuilder(QuasiElastic);
  thePro->RegisterMe(theQGSPPro);
  //  thePro->RegisterMe(theLEPPro=new G4LEPProtonBuilder);
  // theLEPPro->SetMaxEnergy(25*GeV);
  
  thePiK=new G4PiKBuilder_WP;
  theQGSPPiK=new G4QGSPPiKBuilder(QuasiElastic);
  thePiK->RegisterMe(theQGSPPiK);
  // thePiK->RegisterMe(theLEPPiK=new G4LEPPiKBuilder);
  // theLEPPiK->SetMaxEnergy(25*GeV);
  
  theMiscLHEP=new G4MiscLHEPBuilder_WP;
}

HadronPhysicsQGSP_WP::~HadronPhysicsQGSP_WP()
{
   delete theMiscLHEP;
   delete theQGSPNeutron;
   // delete theLEPNeutron;
   delete theQGSPPro;
   //  delete theLEPPro;
   delete thePro;
   delete theQGSPPiK;
   // delete theLEPPiK;
   delete thePiK;
}


void HadronPhysicsQGSP_WP::ConstructParticle()
{
  G4MesonConstructor pMesonConstructor;
  pMesonConstructor.ConstructParticle();

  G4BaryonConstructor pBaryonConstructor;
  pBaryonConstructor.ConstructParticle();

  G4ShortLivedConstructor pShortLivedConstructor;
  pShortLivedConstructor.ConstructParticle();  
}

#include "G4ProcessManager.hh"
void HadronPhysicsQGSP_WP::ConstructProcess()
{
  CreateModels();
  theNeutrons->Build();
  thePro->Build();
  thePiK->Build();
  theMiscLHEP->Build();
}

// Sept 2007 Modified for CMS GflashHadronWrapperProcess

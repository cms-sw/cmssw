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
// GEANT4 tag $Name: CMSSW_6_2_0 $
//
//---------------------------------------------------------------------------
//
// ClassName:   G4MiscLHEPBuilder_WP
//
// Author: 2002 J.P. Wellisch
//
// Modified:
// 16.11.2005 G.Folger: don't  keep processes as data members, but new these
// 13.06.2006 G.Folger: (re)move elastic scatterring 
//
//----------------------------------------------------------------------------
//
#include "SimG4Core/GFlash/interface/GflashHadronWrapperProcess.h"
#include "SimG4Core/GFlash/interface/G4MiscLHEPBuilder_WP.h"

#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

G4MiscLHEPBuilder_WP::G4MiscLHEPBuilder_WP(): wasActivated(false)
{
  theWrappedAntiProtonInelastic=new GflashHadronWrapperProcess("G4AntiProtonInelasticProcess");
}
G4MiscLHEPBuilder_WP::~G4MiscLHEPBuilder_WP()
{
  //  delete theWrappedAntiProtonInelastic;
}

void G4MiscLHEPBuilder_WP::Build()
{
  G4ProcessManager * aProcMan = 0;
  wasActivated = true;
  
  // anti-Proton
  aProcMan = G4AntiProton::AntiProton()->GetProcessManager();
  theLEAntiProtonModel = new G4LEAntiProtonInelastic();
  theHEAntiProtonModel = new G4HEAntiProtonInelastic();
  theHEAntiProtonModel->SetMaxEnergy(100*TeV);
  theAntiProtonInelastic.RegisterMe(theLEAntiProtonModel);
  theAntiProtonInelastic.RegisterMe(theHEAntiProtonModel);
  //  aProcMan->AddDiscreteProcess(&theAntiProtonInelastic);
  edm::LogInfo("SimG4CoreGFlash") << " Adding GflashHadronWrapperProcess (G4wrapperProcess) for G4AntiProtonInelasticProcess" ;
  theWrappedAntiProtonInelastic->RegisterProcess(&theAntiProtonInelastic);
  aProcMan->AddDiscreteProcess(theWrappedAntiProtonInelastic);

  // AntiNeutron
  aProcMan = G4AntiNeutron::AntiNeutron()->GetProcessManager();
  theLEAntiNeutronModel = new G4LEAntiNeutronInelastic();
  theHEAntiNeutronModel = new G4HEAntiNeutronInelastic();
  theHEAntiNeutronModel->SetMaxEnergy(100*TeV);
  theAntiNeutronInelastic.RegisterMe(theLEAntiNeutronModel);
  theAntiNeutronInelastic.RegisterMe(theHEAntiNeutronModel);
  aProcMan->AddDiscreteProcess(&theAntiNeutronInelastic);

  // Lambda
  aProcMan = G4Lambda::Lambda()->GetProcessManager();
  theLELambdaModel = new G4LELambdaInelastic();
  theHELambdaModel = new G4HELambdaInelastic();
  theHELambdaModel->SetMaxEnergy(100*TeV);
  theLambdaInelastic.RegisterMe(theLELambdaModel);
  theLambdaInelastic.RegisterMe(theHELambdaModel);
  aProcMan->AddDiscreteProcess(&theLambdaInelastic);
  
  // AntiLambda
  aProcMan = G4AntiLambda::AntiLambda()->GetProcessManager();
  theLEAntiLambdaModel = new G4LEAntiLambdaInelastic();
  theHEAntiLambdaModel = new G4HEAntiLambdaInelastic();
  theHEAntiLambdaModel->SetMaxEnergy(100*TeV);
  theAntiLambdaInelastic.RegisterMe(theLEAntiLambdaModel);
  theAntiLambdaInelastic.RegisterMe(theHEAntiLambdaModel);
  aProcMan->AddDiscreteProcess(&theAntiLambdaInelastic);
    
  // SigmaMinus
  aProcMan = G4SigmaMinus::SigmaMinus()->GetProcessManager();
  theLESigmaMinusModel = new G4LESigmaMinusInelastic();
  theHESigmaMinusModel = new G4HESigmaMinusInelastic();
  theHESigmaMinusModel->SetMaxEnergy(100*TeV);
  theSigmaMinusInelastic.RegisterMe(theLESigmaMinusModel);
  theSigmaMinusInelastic.RegisterMe(theHESigmaMinusModel);
  aProcMan->AddDiscreteProcess(&theSigmaMinusInelastic);

  // anti-SigmaMinus
  aProcMan = G4AntiSigmaMinus::AntiSigmaMinus()->GetProcessManager();
  theLEAntiSigmaMinusModel = new G4LEAntiSigmaMinusInelastic();
  theHEAntiSigmaMinusModel = new G4HEAntiSigmaMinusInelastic();
  theHEAntiSigmaMinusModel->SetMaxEnergy(100*TeV);
  theAntiSigmaMinusInelastic.RegisterMe(theLEAntiSigmaMinusModel);
  theAntiSigmaMinusInelastic.RegisterMe(theHEAntiSigmaMinusModel);
  aProcMan->AddDiscreteProcess(&theAntiSigmaMinusInelastic);

  // SigmaPlus
  aProcMan = G4SigmaPlus::SigmaPlus()->GetProcessManager();
  theLESigmaPlusModel = new G4LESigmaPlusInelastic();
  theHESigmaPlusModel = new G4HESigmaPlusInelastic();
  theHESigmaPlusModel->SetMaxEnergy(100*TeV);
  theSigmaPlusInelastic.RegisterMe(theLESigmaPlusModel);
  theSigmaPlusInelastic.RegisterMe(theHESigmaPlusModel);
  aProcMan->AddDiscreteProcess(&theSigmaPlusInelastic);

  // anti-SigmaPlus
  aProcMan = G4AntiSigmaPlus::AntiSigmaPlus()->GetProcessManager();
  theLEAntiSigmaPlusModel = new G4LEAntiSigmaPlusInelastic();
  theHEAntiSigmaPlusModel = new G4HEAntiSigmaPlusInelastic();
  theHEAntiSigmaPlusModel->SetMaxEnergy(100*TeV);
  theAntiSigmaPlusInelastic.RegisterMe(theLEAntiSigmaPlusModel);
  theAntiSigmaPlusInelastic.RegisterMe(theHEAntiSigmaPlusModel);
  aProcMan->AddDiscreteProcess(&theAntiSigmaPlusInelastic);

  // XiMinus
  aProcMan = G4XiMinus::XiMinus()->GetProcessManager();
  theLEXiMinusModel = new G4LEXiMinusInelastic();
  theHEXiMinusModel = new G4HEXiMinusInelastic();
  theHEXiMinusModel->SetMaxEnergy(100*TeV);
  theXiMinusInelastic.RegisterMe(theLEXiMinusModel);
  theXiMinusInelastic.RegisterMe(theHEXiMinusModel);
  aProcMan->AddDiscreteProcess(&theXiMinusInelastic);

  // anti-XiMinus
  aProcMan = G4AntiXiMinus::AntiXiMinus()->GetProcessManager();
  theLEAntiXiMinusModel = new G4LEAntiXiMinusInelastic();
  theHEAntiXiMinusModel = new G4HEAntiXiMinusInelastic();
  theHEAntiXiMinusModel->SetMaxEnergy(100*TeV);
  theAntiXiMinusInelastic.RegisterMe(theLEAntiXiMinusModel);
  theAntiXiMinusInelastic.RegisterMe(theHEAntiXiMinusModel);
  aProcMan->AddDiscreteProcess(&theAntiXiMinusInelastic);

  // XiZero
  aProcMan = G4XiZero::XiZero()->GetProcessManager();
  theLEXiZeroModel = new G4LEXiZeroInelastic();
  theHEXiZeroModel = new G4HEXiZeroInelastic();
  theHEXiZeroModel->SetMaxEnergy(100*TeV);
  theXiZeroInelastic.RegisterMe(theLEXiZeroModel);
  theXiZeroInelastic.RegisterMe(theHEXiZeroModel);
  aProcMan->AddDiscreteProcess(&theXiZeroInelastic);

  // anti-XiZero
  aProcMan = G4AntiXiZero::AntiXiZero()->GetProcessManager();
  theLEAntiXiZeroModel = new G4LEAntiXiZeroInelastic();
  theHEAntiXiZeroModel = new G4HEAntiXiZeroInelastic();
  theHEAntiXiZeroModel->SetMaxEnergy(100*TeV);
  theAntiXiZeroInelastic.RegisterMe(theLEAntiXiZeroModel);
  theAntiXiZeroInelastic.RegisterMe(theHEAntiXiZeroModel);
  aProcMan->AddDiscreteProcess(&theAntiXiZeroInelastic);

  // OmegaMinus
  aProcMan = G4OmegaMinus::OmegaMinus()->GetProcessManager();
  theLEOmegaMinusModel = new G4LEOmegaMinusInelastic();
  theHEOmegaMinusModel = new G4HEOmegaMinusInelastic();
  theHEOmegaMinusModel->SetMaxEnergy(100*TeV);
  theOmegaMinusInelastic.RegisterMe(theLEOmegaMinusModel);
  theOmegaMinusInelastic.RegisterMe(theHEOmegaMinusModel);
  aProcMan->AddDiscreteProcess(&theOmegaMinusInelastic);

  // anti-OmegaMinus
  aProcMan = G4AntiOmegaMinus::AntiOmegaMinus()->GetProcessManager();
  theLEAntiOmegaMinusModel = new G4LEAntiOmegaMinusInelastic();
  theHEAntiOmegaMinusModel = new G4HEAntiOmegaMinusInelastic();
  theHEAntiOmegaMinusModel->SetMaxEnergy(100*TeV);
  theAntiOmegaMinusInelastic.RegisterMe(theLEAntiOmegaMinusModel);
  theAntiOmegaMinusInelastic.RegisterMe(theHEAntiOmegaMinusModel);
  aProcMan->AddDiscreteProcess(&theAntiOmegaMinusInelastic);
}

// 2002 by J.P. Wellisch

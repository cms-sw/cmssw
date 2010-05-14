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
// $Id: GflashEMShowerModelMessenger.cc,v 1.2 2007/12/07 23:01:28 dwjang Exp $
// GEANT4 tag $Name: V00-06-02 $
//
//
// ------------------------------------------------------------
// GEANT 4 class implementation
//
//      ------------- GFlashShowerModelMessenger -------------
//
// Author: Joanna Weng - 9.11.2004
// ------------------------------------------------------------

#include "SimG4Core/GFlash/interface/GflashEMShowerModelMessenger.h"
#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"

#include "GFlashParticleBounds.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh" 
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "globals.hh"

#include <iomanip>                
#include <sstream>

GflashEMShowerModelMessenger::
GflashEMShowerModelMessenger(GflashEMShowerModel * aModel)
{ 
  myParaDir = new G4UIdirectory("/GFlash/");
  myParaDir->SetGuidance("Parametrisation control.");
  myModel= aModel;
  
  FlagCmd = new G4UIcmdWithAnInteger("/GFlash/flag",this);
  FlagCmd->SetGuidance("Defines if GFlash is activated");
  FlagCmd->SetParameterName("flag",false,false);
  
  ContCmd = new G4UIcmdWithAnInteger("/GFlash/containment ",this);
  ContCmd->SetGuidance("Defines if Containment is checked");
  ContCmd->SetParameterName("flag",false,false);
  
  StepInX0Cmd = new G4UIcmdWithADouble("/GFlash/stepXo",this);
  StepInX0Cmd->SetGuidance("Defines step lenghts");
  StepInX0Cmd->SetParameterName("flag",false,false);
  
  EminCmd = new G4UIcmdWithADoubleAndUnit("/GFlash/Emin",this);
  EminCmd->SetGuidance("Set minimum kinetic energy to trigger parametrisation");
  EminCmd->SetParameterName("Emin",false,false);
  EminCmd->SetDefaultUnit("GeV");
  EminCmd->SetUnitCategory("Energy");
  EminCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  
  EmaxCmd = new G4UIcmdWithADoubleAndUnit("/GFlash/Emax",this);
  EmaxCmd->SetGuidance("Set maximum kinetic energy to trigger parametrisation");
  EmaxCmd->SetParameterName("Emax",false,false);
  EmaxCmd->SetDefaultUnit("GeV");
  EmaxCmd->SetUnitCategory("Energy");
  EmaxCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
  
  EkillCmd = new G4UIcmdWithADoubleAndUnit("/GFlash/Ekill",this);
  EkillCmd->SetGuidance("Set maximum kinetic energy for electrons to be killed");
  EkillCmd->SetParameterName("Ekill",false,false);
  EkillCmd->SetDefaultUnit("GeV");
  EkillCmd->SetUnitCategory("Energy");
  EkillCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
}


GflashEMShowerModelMessenger::~GflashEMShowerModelMessenger()
{
  delete ContCmd;
  delete FlagCmd;
  delete StepInX0Cmd;  
  delete EminCmd;
  delete EmaxCmd;
  delete EkillCmd;
}


void GflashEMShowerModelMessenger::
SetNewValue(G4UIcommand * command,G4String newValues)
{ 
  /*  
  if( command == FlagCmd ) { 
    myModel->SetFlagParamType(FlagCmd->GetNewIntValue(newValues));      
    this->GetCurrentValue(command);    
  }
  if( command == ContCmd ) { 
    myModel->SetFlagParticleContainment(ContCmd->GetNewIntValue(newValues));      
    this->GetCurrentValue(command);    
  }
  if( command == StepInX0Cmd ) { 
    myModel->SetStepInX0(StepInX0Cmd->GetNewDoubleValue(newValues));      
    this->GetCurrentValue(command);    
  }
  
  else if( command == EminCmd ) {
    myModel->PBound->SetMinEneToParametrise(*G4Electron::ElectronDefinition(),
                                       EminCmd->GetNewDoubleValue(newValues));
    this->GetCurrentValue(command);  
  }
  
  else if( command == EmaxCmd ) {
    myModel->PBound->SetMaxEneToParametrise(*G4Electron::ElectronDefinition(),
                                       EmaxCmd->GetNewDoubleValue(newValues));
    this->GetCurrentValue(command);      
  }
  
  else if( command == EkillCmd ) {
    myModel->PBound->SetEneToKill(*G4Electron::ElectronDefinition(),
                                       EkillCmd->GetNewDoubleValue(newValues));
    this->GetCurrentValue(command);  
  }
  */
}


G4String GflashEMShowerModelMessenger::GetCurrentValue(G4UIcommand * command)
{
  G4String returnValue('\0');
  std::ostringstream os;
  /*  
  if( command == FlagCmd ) { 
    os << "/GFlash/flag " << myModel->GetFlagParamType()  << '\0';
    returnValue = G4String(os.str());
  }
  
  else if( command == EkillCmd ) {    
    os << "/GFlash/Ekill "
       << myModel->PBound->GetEneToKill(*G4Electron::ElectronDefinition())/GeV
       << " GeV" << '\0';
    returnValue = G4String(os.str());
  }
  
  else if( command == EminCmd ) {    
    os << "/GFlash/Emin "
       << myModel->PBound->GetMinEneToParametrise(*G4Electron::ElectronDefinition())/GeV
       << " GeV" << '\0';
    returnValue = G4String(os.str());  
  }
  
  else if( command == EmaxCmd ) {
    os << "/GFlash/Emax "
       << myModel->PBound->GetMaxEneToParametrise(*G4Electron::ElectronDefinition())/GeV
       << " GeV" << '\0';
    returnValue = G4String(os.str());
  }
  */
  return returnValue;
}

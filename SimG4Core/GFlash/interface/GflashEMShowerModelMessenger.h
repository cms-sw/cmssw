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
// $Id: GflashEMShowerModelMessenger.h,v 1.1 2007/05/15 23:16:40 syjun Exp $
// GEANT4 tag $Name: V00-06-02 $
//
//
//---------------------------------------------------------------
//  GEANT 4 class header file
//
//  GflashEMShowerModelMessenger
//
//  Class description:
//
//  Messenger for the GFlash parameterisation shower model control.

//
// Author: Joanna Weng - 9.11.04
//---------------------------------------------------------------

#ifndef GflashEMShowerModelMessenger_h
#define GflashEMShowerModelMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class GflashEMShowerModel;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithADouble;

class GflashEMShowerModelMessenger: public G4UImessenger
{
  public:

    GflashEMShowerModelMessenger(GflashEMShowerModel * myModel);
    ~GflashEMShowerModelMessenger();
  
    void SetNewValue(G4UIcommand * command,G4String newValues);
    G4String GetCurrentValue(G4UIcommand * command);
  
  private:

    GflashEMShowerModel* myModel;
    G4UIdirectory*   myParaDir;
    G4UIcmdWithAString*  SwitchCmd;
    G4UIcmdWithAnInteger*  FlagCmd;
    G4UIcmdWithAnInteger*  ContCmd; // Containment Check
    G4UIcmdWithADouble*   StepInX0Cmd;
    G4UIcmdWithADoubleAndUnit*   EmaxCmd;
    G4UIcmdWithADoubleAndUnit*   EminCmd;
    G4UIcmdWithADoubleAndUnit*   EkillCmd;
};

#endif

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
// $Id:$
// GEANT4 tag $Name: not supported by cvs2svn $
//
// --------------------------------------------------------------
//      GEANT 4 class implementation file
//
//  G4Physics2DVectorCache.cc
//
//  Author:        Vladimir Ivanchenko 
//                 on base of  Hisaya Kurashige 1D class
//
//  Creation date: 25.09.2011
// --------------------------------------------------------------

#include "G4Physics2DVectorCache95.hh"


G4Physics2DVectorCache95::G4Physics2DVectorCache95()
{
  Clear();
}

G4Physics2DVectorCache95::~G4Physics2DVectorCache95()
{
}

void G4Physics2DVectorCache95::Clear()
{
  lastX = lastY = lastValue = 0.0;
  lastBinX = lastBinY = 0;  
}

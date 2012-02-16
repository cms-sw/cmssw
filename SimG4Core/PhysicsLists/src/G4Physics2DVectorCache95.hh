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
// 
// ---------------------------------------------------------------
// GEANT 4 class header file
//
// G4Physics2DVectorCache95
//
// Class description:
//
// This class includes cache data in use by G4Physics2DVector.

//  Author:        Vladimir Ivanchenko 
//                 on base of  Hisaya Kurashige 1D class
//
//  Creation date: 25.09.2011
// ---------------------------------------------------------------

#ifndef G4Physics2DVectorCache95_h
#define G4Physics2DVectorCache95_h 1

#include "globals.hh"

class G4Physics2DVectorCache95 
{
public:  

  G4Physics2DVectorCache95();

  ~G4Physics2DVectorCache95();

  void Clear();

  G4double lastX;
  G4double lastY;
  G4double lastValue;
  size_t   lastBinX;
  size_t   lastBinY;
};


#endif

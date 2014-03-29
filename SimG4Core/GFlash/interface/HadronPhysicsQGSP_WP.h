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
// GEANT4 tag $Name: V04-05-01 $
//
//---------------------------------------------------------------------------
//
// ClassName:   HadronPhysicsQGSP_WP
//
// Author: 2002 J.P. Wellisch
//
// Modified:
// 21.11.2005 G.Folger: don't  keep processes as data members, but new these
// 08.06.2006 V.Ivanchenko: remove stopping
// 30.03.2007 G.Folger: Add quasielastic option, turned off by default
// 25.04.2007 G.Folger: Use quasielastic by default
//
//----------------------------------------------------------------------------
//
#ifndef HadronPhysicsQGSP_WP_h
#define HadronPhysicsQGSP_WP_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
//#include "G4MiscLHEPBuilder.hh"

//#include "G4LEPPiKBuilder.hh"
#include "G4QGSPPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
//#include "G4LEPProtonBuilder.hh"
#include "G4QGSPProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
//#include "G4LEPNeutronBuilder.hh"
#include "G4QGSPNeutronBuilder.hh"

class G4PiKBuilder_WP;
class G4ProtonBuilder_WP;
class G4MiscLHEPBuilder_WP;

class HadronPhysicsQGSP_WP : public G4VPhysicsConstructor
{
  public: 
    HadronPhysicsQGSP_WP(const G4String& name ="hadron",G4bool quasiElastic=true);
    virtual ~HadronPhysicsQGSP_WP();

  public: 
    virtual void ConstructParticle();
    virtual void ConstructProcess();

  private:
    void CreateModels();
    G4NeutronBuilder * theNeutrons;
    //G4LEPNeutronBuilder * theLEPNeutron;
    G4QGSPNeutronBuilder * theQGSPNeutron;
    
    G4PiKBuilder_WP * thePiK;
    //G4LEPPiKBuilder * theLEPPiK;
    G4QGSPPiKBuilder * theQGSPPiK;
    
    G4ProtonBuilder_WP * thePro;
    // G4LEPProtonBuilder * theLEPPro;
    G4QGSPProtonBuilder * theQGSPPro;    
    
    G4MiscLHEPBuilder_WP * theMiscLHEP;
    
    G4bool QuasiElastic;
};
// 2008 Modified for CMS GflashHadronWrapperProcess
#endif


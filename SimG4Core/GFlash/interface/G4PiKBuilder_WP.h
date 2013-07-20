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
// $Id: G4PiKBuilder_WP.h,v 1.4 2013/05/30 21:10:49 gartung Exp $
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
#ifndef G4PiKBuilder_WP_h
#define G4PiKBuilder_WP_h 1

#include "globals.hh"

#include "G4ProtonInelasticProcess.hh"
#include "G4VPiKBuilder.hh"

#include <vector>

class GflashHadronWrapperProcess;

class G4PiKBuilder_WP
{
  public: 
    G4PiKBuilder_WP();
    virtual ~G4PiKBuilder_WP();

  public: 
    void Build();
    void RegisterMe(G4VPiKBuilder * aB) {theModelCollections.push_back(aB);}

  private:
    G4PionPlusInelasticProcess*  thePionPlusInelastic;
    G4PionMinusInelasticProcess* thePionMinusInelastic;
    G4KaonPlusInelasticProcess*  theKaonPlusInelastic;
    G4KaonMinusInelasticProcess* theKaonMinusInelastic;
    G4KaonZeroLInelasticProcess* theKaonZeroLInelastic;
    G4KaonZeroSInelasticProcess* theKaonZeroSInelastic;
     
    GflashHadronWrapperProcess*  theWrappedPionPlusInelastic;
    GflashHadronWrapperProcess*  theWrappedPionMinusInelastic;
    GflashHadronWrapperProcess*  theWrappedKaonPlusInelastic;
    GflashHadronWrapperProcess*  theWrappedKaonMinusInelastic;

    std::vector<G4VPiKBuilder *> theModelCollections;

    G4bool wasActivated;
};

// 2002 by J.P. Wellisch
// 2008 Modified for CMS GflashHadronWrapperProcess
#endif


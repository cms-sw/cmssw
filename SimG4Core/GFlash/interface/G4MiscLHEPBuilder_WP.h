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
#ifndef G4MiscLHEPBuilder_WP_h
#define G4MiscLHEPBuilder_WP_h 1

#include "globals.hh"

#include "G4AntiProtonInelasticProcess.hh"
#include "G4AntiNeutronInelasticProcess.hh"
#include "G4LambdaInelasticProcess.hh"
#include "G4AntiLambdaInelasticProcess.hh"
#include "G4SigmaPlusInelasticProcess.hh"
#include "G4SigmaMinusInelasticProcess.hh"
#include "G4AntiSigmaPlusInelasticProcess.hh"
#include "G4AntiSigmaMinusInelasticProcess.hh"
#include "G4XiZeroInelasticProcess.hh"
#include "G4XiMinusInelasticProcess.hh"
#include "G4AntiXiZeroInelasticProcess.hh"
#include "G4AntiXiMinusInelasticProcess.hh"
#include "G4OmegaMinusInelasticProcess.hh"
#include "G4AntiOmegaMinusInelasticProcess.hh"

#include "G4LEAntiProtonInelastic.hh"
#include "G4LEAntiNeutronInelastic.hh"
#include "G4LELambdaInelastic.hh"
#include "G4LEAntiLambdaInelastic.hh"
#include "G4LESigmaPlusInelastic.hh"
#include "G4LESigmaMinusInelastic.hh"
#include "G4LEAntiSigmaPlusInelastic.hh"
#include "G4LEAntiSigmaMinusInelastic.hh"
#include "G4LEXiZeroInelastic.hh"
#include "G4LEXiMinusInelastic.hh"
#include "G4LEAntiXiZeroInelastic.hh"
#include "G4LEAntiXiMinusInelastic.hh"
#include "G4LEOmegaMinusInelastic.hh"
#include "G4LEAntiOmegaMinusInelastic.hh"

// High-energy Models

#include "G4HEAntiProtonInelastic.hh"
#include "G4HEAntiNeutronInelastic.hh"
#include "G4HELambdaInelastic.hh"
#include "G4HEAntiLambdaInelastic.hh"
#include "G4HESigmaPlusInelastic.hh"
#include "G4HESigmaMinusInelastic.hh"
#include "G4HEAntiSigmaPlusInelastic.hh"
#include "G4HEAntiSigmaMinusInelastic.hh"
#include "G4HEXiZeroInelastic.hh"
#include "G4HEXiMinusInelastic.hh"
#include "G4HEAntiXiZeroInelastic.hh"
#include "G4HEAntiXiMinusInelastic.hh"
#include "G4HEOmegaMinusInelastic.hh"
#include "G4HEAntiOmegaMinusInelastic.hh"

class GflashHadronWrapperProcess;

class G4MiscLHEPBuilder_WP 
{
  public: 
    G4MiscLHEPBuilder_WP();
    virtual ~G4MiscLHEPBuilder_WP();

  public: 
    void Build();

  private:
 
    // anti-proton
    G4AntiProtonInelasticProcess theAntiProtonInelastic;
    G4LEAntiProtonInelastic* theLEAntiProtonModel;
    G4HEAntiProtonInelastic* theHEAntiProtonModel;

    // anti-proton wrapper process for Gflash 
    GflashHadronWrapperProcess*  theWrappedAntiProtonInelastic;

    // anti-neutron
    G4AntiNeutronInelasticProcess  theAntiNeutronInelastic;
    G4LEAntiNeutronInelastic* theLEAntiNeutronModel;
    G4HEAntiNeutronInelastic* theHEAntiNeutronModel;

    // Lambda
    G4LambdaInelasticProcess  theLambdaInelastic;
    G4LELambdaInelastic*  theLELambdaModel;
    G4HELambdaInelastic*  theHELambdaModel;

    // AntiLambda
    G4AntiLambdaInelasticProcess  theAntiLambdaInelastic;
    G4LEAntiLambdaInelastic*  theLEAntiLambdaModel;
    G4HEAntiLambdaInelastic*  theHEAntiLambdaModel;

    // SigmaMinus
    G4SigmaMinusInelasticProcess  theSigmaMinusInelastic;
    G4LESigmaMinusInelastic*  theLESigmaMinusModel;
    G4HESigmaMinusInelastic*  theHESigmaMinusModel;

    // AntiSigmaMinus
    G4AntiSigmaMinusInelasticProcess  theAntiSigmaMinusInelastic;
    G4LEAntiSigmaMinusInelastic*  theLEAntiSigmaMinusModel;
    G4HEAntiSigmaMinusInelastic*  theHEAntiSigmaMinusModel;

    // SigmaPlus
    G4SigmaPlusInelasticProcess  theSigmaPlusInelastic;
    G4LESigmaPlusInelastic*  theLESigmaPlusModel;
    G4HESigmaPlusInelastic*  theHESigmaPlusModel;

    // AntiSigmaPlus
    G4AntiSigmaPlusInelasticProcess  theAntiSigmaPlusInelastic;
    G4LEAntiSigmaPlusInelastic*  theLEAntiSigmaPlusModel;
    G4HEAntiSigmaPlusInelastic*  theHEAntiSigmaPlusModel;

    // XiZero
    G4XiZeroInelasticProcess  theXiZeroInelastic;
    G4LEXiZeroInelastic*  theLEXiZeroModel;
    G4HEXiZeroInelastic*  theHEXiZeroModel;

    // AntiXiZero
    G4AntiXiZeroInelasticProcess  theAntiXiZeroInelastic;
    G4LEAntiXiZeroInelastic*  theLEAntiXiZeroModel;
    G4HEAntiXiZeroInelastic*  theHEAntiXiZeroModel;

    // XiMinus
    G4XiMinusInelasticProcess  theXiMinusInelastic;
    G4LEXiMinusInelastic*  theLEXiMinusModel;
    G4HEXiMinusInelastic*  theHEXiMinusModel;

    // AntiXiMinus
    G4AntiXiMinusInelasticProcess  theAntiXiMinusInelastic;
    G4LEAntiXiMinusInelastic*  theLEAntiXiMinusModel;
    G4HEAntiXiMinusInelastic*  theHEAntiXiMinusModel;

    // OmegaMinus
    G4OmegaMinusInelasticProcess  theOmegaMinusInelastic;
    G4LEOmegaMinusInelastic*  theLEOmegaMinusModel;
    G4HEOmegaMinusInelastic*  theHEOmegaMinusModel;

    // AntiOmegaMinus
    G4AntiOmegaMinusInelasticProcess  theAntiOmegaMinusInelastic;
    G4LEAntiOmegaMinusInelastic*  theLEAntiOmegaMinusModel;
    G4HEAntiOmegaMinusInelastic*  theHEAntiOmegaMinusModel;

    G4bool wasActivated;
};
// 2002 by J.P. Wellisch

#endif

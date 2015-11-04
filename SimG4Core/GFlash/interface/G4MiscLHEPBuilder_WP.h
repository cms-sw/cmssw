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
    //    G4LEAntiProtonInelastic* theLEAntiProtonModel;
    //  G4HEAntiProtonInelastic* theHEAntiProtonModel;

    // anti-proton wrapper process for Gflash 
    GflashHadronWrapperProcess*  theWrappedAntiProtonInelastic;

    // anti-neutron
    G4AntiNeutronInelasticProcess  theAntiNeutronInelastic;
    // G4LEAntiNeutronInelastic* theLEAntiNeutronModel;
    // G4HEAntiNeutronInelastic* theHEAntiNeutronModel;

    // Lambda
    G4LambdaInelasticProcess  theLambdaInelastic;
    // G4LELambdaInelastic*  theLELambdaModel;
    // G4HELambdaInelastic*  theHELambdaModel;

    // AntiLambda
    G4AntiLambdaInelasticProcess  theAntiLambdaInelastic;
    // G4LEAntiLambdaInelastic*  theLEAntiLambdaModel;
    // G4HEAntiLambdaInelastic*  theHEAntiLambdaModel;

    // SigmaMinus
    G4SigmaMinusInelasticProcess  theSigmaMinusInelastic;
    // G4LESigmaMinusInelastic*  theLESigmaMinusModel;
    // G4HESigmaMinusInelastic*  theHESigmaMinusModel;

    // AntiSigmaMinus
    G4AntiSigmaMinusInelasticProcess  theAntiSigmaMinusInelastic;
    // G4LEAntiSigmaMinusInelastic*  theLEAntiSigmaMinusModel;
    // G4HEAntiSigmaMinusInelastic*  theHEAntiSigmaMinusModel;

    // SigmaPlus
    G4SigmaPlusInelasticProcess  theSigmaPlusInelastic;
    // G4LESigmaPlusInelastic*  theLESigmaPlusModel;
    // G4HESigmaPlusInelastic*  theHESigmaPlusModel;

    // AntiSigmaPlus
    G4AntiSigmaPlusInelasticProcess  theAntiSigmaPlusInelastic;
    // G4LEAntiSigmaPlusInelastic*  theLEAntiSigmaPlusModel;
    // G4HEAntiSigmaPlusInelastic*  theHEAntiSigmaPlusModel;

    // XiZero
    G4XiZeroInelasticProcess  theXiZeroInelastic;
    // G4LEXiZeroInelastic*  theLEXiZeroModel;
    // G4HEXiZeroInelastic*  theHEXiZeroModel;

    // AntiXiZero
    G4AntiXiZeroInelasticProcess  theAntiXiZeroInelastic;
    // G4LEAntiXiZeroInelastic*  theLEAntiXiZeroModel;
    // G4HEAntiXiZeroInelastic*  theHEAntiXiZeroModel;

    // XiMinus
    G4XiMinusInelasticProcess  theXiMinusInelastic;
    // G4LEXiMinusInelastic*  theLEXiMinusModel;
    // G4HEXiMinusInelastic*  theHEXiMinusModel;

    // AntiXiMinus
    G4AntiXiMinusInelasticProcess  theAntiXiMinusInelastic;
    // G4LEAntiXiMinusInelastic*  theLEAntiXiMinusModel;
    // G4HEAntiXiMinusInelastic*  theHEAntiXiMinusModel;

    // OmegaMinus
    G4OmegaMinusInelasticProcess  theOmegaMinusInelastic;
    // G4LEOmegaMinusInelastic*  theLEOmegaMinusModel;
    // G4HEOmegaMinusInelastic*  theHEOmegaMinusModel;

    // AntiOmegaMinus
    G4AntiOmegaMinusInelasticProcess  theAntiOmegaMinusInelastic;
    // G4LEAntiOmegaMinusInelastic*  theLEAntiOmegaMinusModel;
    // G4HEAntiOmegaMinusInelastic*  theHEAntiOmegaMinusModel;

    G4bool wasActivated;
};
// 2002 by J.P. Wellisch

#endif

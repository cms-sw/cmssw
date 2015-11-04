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


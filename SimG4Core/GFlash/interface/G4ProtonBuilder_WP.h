//
//---------------------------------------------------------------------------
//
// ClassName:   G4ProtonBuilder_WP
//
// Author: 2002 J.P. Wellisch
//
// Modified:
// 16.11.2005 G.Folger: don't  keep processes as data members, but new these
// 13.06.2006 G.Folger: (re)move elastic scatterring 
//
//----------------------------------------------------------------------------
//
#ifndef G4ProtonBuilder_WP_h
#define G4ProtonBuilder_WP_h 1

#include "globals.hh"

#include "G4ProtonInelasticProcess.hh"
#include "G4VProtonBuilder.hh"

#include <vector>

class GflashHadronWrapperProcess;

class G4ProtonBuilder_WP
{
  public: 
    G4ProtonBuilder_WP();
    virtual ~G4ProtonBuilder_WP();

  public: 
    void Build();
    void RegisterMe(G4VProtonBuilder * aB) {theModelCollections.push_back(aB);}

  private:
    G4ProtonInelasticProcess * theProtonInelastic;

    GflashHadronWrapperProcess*  theWrappedProtonInelastic;
    
    std::vector<G4VProtonBuilder *> theModelCollections;

    G4bool wasActivated;
};

// 2002 by J.P. Wellisch
// 2009 Modified for CMS GflashHadronWrapperProcess
#endif


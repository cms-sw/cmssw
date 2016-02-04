#ifndef G4RPGProtonBuilder_h
#define G4RPGProtonBuilder_h 1

#include "globals.hh"

#include "G4HadronElasticProcess.hh"
#include "G4ProtonInelasticProcess.hh"
#include "G4VProtonBuilder.hh"

#include "G4RPGProtonInelastic.hh"

class G4RPGProtonBuilder : public G4VProtonBuilder
{
  public: 
    G4RPGProtonBuilder();
    virtual ~G4RPGProtonBuilder();

  public: 
    virtual void Build(G4ProtonInelasticProcess * aP);
    virtual void Build(G4HadronElasticProcess * aP);
    
    void SetMinEnergy(G4double aM) 
    {
      theMin=aM;
    }
    void SetMaxEnergy(G4double aM) 
    {
      theMax=aM;
    }
    
  private:
    G4double theMin;
    G4double theMax;
    G4RPGProtonInelastic * theRPGProtonModel;

};

// 2002 by J.P. Wellisch

#endif


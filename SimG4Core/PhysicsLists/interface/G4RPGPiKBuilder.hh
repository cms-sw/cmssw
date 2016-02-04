#ifndef G4RPGPiKBuilder_h
#define G4RPGPiKBuilder_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPiKBuilder.hh"

#include "G4RPGPiPlusInelastic.hh"
#include "G4RPGPiMinusInelastic.hh"
#include "G4RPGKPlusInelastic.hh"
#include "G4RPGKShortInelastic.hh"
#include "G4RPGKLongInelastic.hh"
#include "G4RPGKMinusInelastic.hh"

class G4RPGPiKBuilder : public G4VPiKBuilder
{
  public: 
    G4RPGPiKBuilder();
    virtual ~G4RPGPiKBuilder();

  public: 
    virtual void Build(G4HadronElasticProcess * aP);
    virtual void Build(G4PionPlusInelasticProcess * aP);
    virtual void Build(G4PionMinusInelasticProcess * aP);
    virtual void Build(G4KaonPlusInelasticProcess * aP);
    virtual void Build(G4KaonMinusInelasticProcess * aP);
    virtual void Build(G4KaonZeroLInelasticProcess * aP);
    virtual void Build(G4KaonZeroSInelasticProcess * aP);
    
    void SetMinEnergy(G4double aM) {theMin = aM;}
    void SetMaxEnergy(G4double aM) {theMax = aM;}

  private:
    G4double theMin;
    G4double theMax;

    G4RPGPiPlusInelastic*  theRPGPiPlusModel;
    G4RPGPiMinusInelastic* theRPGPiMinusModel;
    G4RPGKPlusInelastic*   theRPGKPlusModel;
    G4RPGKMinusInelastic*  theRPGKMinusModel;
    G4RPGKLongInelastic*   theRPGKLongModel;
    G4RPGKShortInelastic*  theRPGKShortModel;

};

#endif


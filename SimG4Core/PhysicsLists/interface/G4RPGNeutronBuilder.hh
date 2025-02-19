#ifndef G4RPGNeutronBuilder_h
#define G4RPGNeutronBuilder_h 1

#include "globals.hh"

#include "G4HadronElasticProcess.hh"
#include "G4HadronFissionProcess.hh"
#include "G4HadronCaptureProcess.hh"
#include "G4NeutronInelasticProcess.hh"
#include "G4VNeutronBuilder.hh"

#include "G4LFission.hh"
#include "G4LCapture.hh"
#include "G4RPGNeutronInelastic.hh"

class G4RPGNeutronBuilder : public G4VNeutronBuilder
{
  public: 
    G4RPGNeutronBuilder();
    virtual ~G4RPGNeutronBuilder();

  public: 
    virtual void Build(G4HadronElasticProcess * aP);
    virtual void Build(G4HadronFissionProcess * aP);
    virtual void Build(G4HadronCaptureProcess * aP);
    virtual void Build(G4NeutronInelasticProcess * aP);
    
    void SetMinEnergy(G4double aM) 
    {
      theMin=aM;
      theIMin = theMin;
    }
    void SetMinInelasticEnergy(G4double aM) 
    {
      theIMin=aM;
    }
    void SetMaxEnergy(G4double aM) 
    {
      theIMax = aM;
      theMax=aM;
    }
    void SetMaxInelasticEnergy(G4double aM)
    {
      theIMax = aM;
    }
    
  private:
    G4double theMin;
    G4double theIMin;
    G4double theMax;
    G4double theIMax;

    G4RPGNeutronInelastic * theRPGNeutronModel;
    G4LFission * theNeutronFissionModel;
    G4LCapture * theNeutronCaptureModel;

};

#endif


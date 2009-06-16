#ifndef SimG4Core_PhysicsLists_CMSFTFPProtonBuilder_h
#define SimG4Core_PhysicsLists_CMSFTFPProtonBuilder_h 

#include "globals.hh"

#include "G4HadronElasticProcess.hh"
#include "G4HadronFissionProcess.hh"
#include "G4HadronCaptureProcess.hh"
#include "G4ProtonInelasticProcess.hh"
#include "G4VProtonBuilder.hh"

#include "G4TheoFSGenerator.hh"
#include "G4ExcitationHandler.hh"
#include "G4PreCompoundModel.hh"
#include "G4GeneratorPrecompoundInterface.hh"
#include "G4FTFModel.hh"
#include "G4LundStringFragmentation.hh"
#include "G4ExcitedStringDecay.hh"
#include "G4QuasiElasticChannel.hh"

class CMSFTFPProtonBuilder : public G4VProtonBuilder
{
  public: 
    CMSFTFPProtonBuilder(G4bool quasiElastic=false);
    virtual ~CMSFTFPProtonBuilder();

  public: 
    virtual void Build(G4HadronElasticProcess * aP);
    virtual void Build(G4ProtonInelasticProcess * aP);
    
    void SetMinEnergy(G4double aM) {theMin = aM;}
    void SetMaxEnergy(G4double aM) {theMax = aM;}

  private:
    G4TheoFSGenerator * theModel;
    G4PreCompoundModel * thePreEquilib;
    G4GeneratorPrecompoundInterface * theCascade;
    G4FTFModel * theStringModel;
    G4ExcitedStringDecay * theStringDecay;
    G4QuasiElasticChannel * theQuasiElastic;

    G4double theMin;
    G4double theMax;

};

// 2002 by J.P. Wellisch

#endif


#ifndef HadronPhysicsQGSC_h
#define HadronPhysicsQGSC_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
#include "G4MiscLHEPBuilder.hh"

#include "G4PiKBuilder.hh"
#include "G4LEPPiKBuilder.hh"
#include "G4QGSCPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4LEPProtonBuilder.hh"
#include "G4QGSCProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4LEPNeutronBuilder.hh"
#include "G4QGSCNeutronBuilder.hh"

class HadronPhysicsQGSC : public G4VPhysicsConstructor
{
  public: 
    HadronPhysicsQGSC(G4int verbose =1);
    HadronPhysicsQGSC(const G4String& name,G4bool quasiElastic=true);
    virtual ~HadronPhysicsQGSC();

  public: 
    virtual void ConstructParticle();
    virtual void ConstructProcess();

  private:
    void CreateModels();
    G4NeutronBuilder * theNeutrons;
    G4LEPNeutronBuilder * theLEPNeutron;
    G4QGSCNeutronBuilder * theQGSCNeutron;
    
    G4PiKBuilder * thePiK;
    G4LEPPiKBuilder * theLEPPiK;
    G4QGSCPiKBuilder * theQGSCPiK;
    
    G4ProtonBuilder * thePro;
    G4LEPProtonBuilder * theLEPPro;
    G4QGSCProtonBuilder * theQGSCPro;    
    
    G4MiscLHEPBuilder * theMiscLHEP;
    
    G4bool QuasiElastic;
};

#endif


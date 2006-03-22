#ifndef HadronPhysicsQGSP_h
#define HadronPhysicsQGSP_h 1

#include "G4VPhysicsConstructor.hh"

#include "SimG4Core/Packaging/src/G4HadronQEDBuilder.hh"
#include "SimG4Core/Packaging/src/G4StoppingHadronBuilder.hh"
#include "SimG4Core/Packaging/src/G4MiscLHEPBuilder.hh"

#include "SimG4Core/Packaging/src/G4PiKBuilder.hh"
#include "SimG4Core/Packaging/src/G4LEPPiKBuilder.hh"
#include "SimG4Core/Packaging/src/G4QGSPPiKBuilder.hh"

#include "SimG4Core/Packaging/src/G4ProtonBuilder.hh"
#include "SimG4Core/Packaging/src/G4LEPProtonBuilder.hh"
#include "SimG4Core/Packaging/src/G4QGSPProtonBuilder.hh"

#include "SimG4Core/Packaging/src/G4NeutronBuilder.hh"
#include "SimG4Core/Packaging/src/G4LEPNeutronBuilder.hh"
#include "SimG4Core/Packaging/src/G4QGSPNeutronBuilder.hh"

class HadronPhysicsQGSP : public G4VPhysicsConstructor
{
  public: 
    HadronPhysicsQGSP(const G4String& name ="hadron");
    virtual ~HadronPhysicsQGSP();

  public: 
    virtual void ConstructParticle();
    virtual void ConstructProcess();

  private:
    G4NeutronBuilder theNeutrons;
    G4LEPNeutronBuilder theLEPNeutron;
    G4QGSPNeutronBuilder theQGSPNeutron;
    
    G4PiKBuilder thePiK;
    G4LEPPiKBuilder theLEPPiK;
    G4QGSPPiKBuilder theQGSPPiK;
    
    G4ProtonBuilder thePro;
    G4LEPProtonBuilder theLEPPro;
    G4QGSPProtonBuilder theQGSPPro;    
    
    G4MiscLHEPBuilder theMiscLHEP;
    G4StoppingHadronBuilder theStoppingHadron;
    G4HadronQEDBuilder theHadronQED;
};

#endif


#ifndef SimG4Core_PhysicsLists_HadronPhysicsCMS_h
#define SimG4Core_PhysicsLists_HadronPhysicsCMS_h 1

#include "globals.hh"
#include "G4ios.hh"

#include "G4VPhysicsConstructor.hh"
#include "G4MiscLHEPBuilder.hh"

#include "G4PiKBuilder.hh"
#include "G4BertiniPiKBuilder.hh"
#include "G4BinaryPiKBuilder.hh"
#include "G4FTFCPiKBuilder.hh"
#include "G4FTFPPiKBuilder.hh"
#include "G4LEPPiKBuilder.hh"
#include "G4LHEPPiKBuilder.hh"
#include "G4QGSCEflowPiKBuilder.hh"
#include "G4QGSCPiKBuilder.hh"
#include "G4QGSPPiKBuilder.hh"

#include "G4ProtonBuilder.hh"
#include "G4BertiniProtonBuilder.hh"
#include "G4BinaryProtonBuilder.hh"
#include "G4FTFCProtonBuilder.hh"
#include "G4FTFPProtonBuilder.hh"
#include "G4LEPProtonBuilder.hh"
#include "G4LHEPProtonBuilder.hh"
#include "G4PrecoProtonBuilder.hh"
#include "G4QGSCEflowProtonBuilder.hh"
#include "G4QGSCProtonBuilder.hh"
#include "G4QGSPProtonBuilder.hh"

#include "G4NeutronBuilder.hh"
#include "G4BertiniNeutronBuilder.hh"
#include "G4BinaryNeutronBuilder.hh"
#include "G4FTFCNeutronBuilder.hh"
#include "G4FTFPNeutronBuilder.hh"
#include "G4LEPNeutronBuilder.hh"
#include "G4LHEPNeutronBuilder.hh"
#include "G4PrecoNeutronBuilder.hh"
#include "G4QGSCEflowNeutronBuilder.hh"
#include "G4QGSCNeutronBuilder.hh"
#include "G4QGSPNeutronBuilder.hh"

#include "G4FTFBinaryNeutronBuilder.hh"  
#include "G4FTFBinaryPiKBuilder.hh"
#include "G4FTFBinaryProtonBuilder.hh"

#include "SimG4Core/PhysicsLists/interface/G4RPGNeutronBuilder.hh"
#include "SimG4Core/PhysicsLists/interface/G4RPGPiKBuilder.hh"
#include "SimG4Core/PhysicsLists/interface/G4RPGProtonBuilder.hh"

class HadronPhysicsCMS : public G4VPhysicsConstructor {

public: 

  HadronPhysicsCMS(const G4String& name ="QGSP", G4bool quasiElastic=true);
  virtual ~HadronPhysicsCMS();

public: 

  virtual void ConstructParticle();
  virtual void ConstructProcess();

private:

  void CreateModels();

  G4NeutronBuilder          * theNeutrons;
  G4BertiniNeutronBuilder   * theBertiniNeutron;
  G4BinaryNeutronBuilder    * theBinaryNeutron;
  G4FTFCNeutronBuilder      * theFTFCNeutron;
  G4FTFPNeutronBuilder      * theFTFPNeutron;
  G4LEPNeutronBuilder       * theLEPNeutron;
  G4LHEPNeutronBuilder      * theLHEPNeutron;
  G4PrecoNeutronBuilder     * thePrecoNeutron;
  G4QGSCEflowNeutronBuilder * theQGSCEflowNeutron;
  G4QGSCNeutronBuilder      * theQGSCNeutron;
  G4QGSPNeutronBuilder      * theQGSPNeutron;
    
  G4PiKBuilder              * thePiK;
  G4BertiniPiKBuilder       * theBertiniPiK;
  G4BinaryPiKBuilder        * theBinaryPiK;
  G4FTFCPiKBuilder          * theFTFCPiK;
  G4FTFPPiKBuilder          * theFTFPPiK;
  G4LEPPiKBuilder           * theLEPPiK;
  G4LHEPPiKBuilder          * theLHEPPiK;
  G4QGSCEflowPiKBuilder     * theQGSCEflowPiK;
  G4QGSCPiKBuilder          * theQGSCPiK;
  G4QGSPPiKBuilder          * theQGSPPiK;
    
  G4ProtonBuilder           * thePro;
  G4BertiniProtonBuilder    * theBertiniPro;
  G4BinaryProtonBuilder     * theBinaryPro;
  G4FTFCProtonBuilder       * theFTFCPro;
  G4FTFPProtonBuilder       * theFTFPPro;
  G4LEPProtonBuilder        * theLEPPro;
  G4LHEPProtonBuilder       * theLHEPPro;
  G4PrecoProtonBuilder      * thePrecoPro;
  G4QGSCEflowProtonBuilder  * theQGSCEflowPro;    
  G4QGSCProtonBuilder       * theQGSCPro;    
  G4QGSPProtonBuilder       * theQGSPPro;    
    
  G4MiscLHEPBuilder         * theMiscLHEP;

  G4FTFBinaryNeutronBuilder * theFTFNeutron;
  G4FTFBinaryPiKBuilder     * theFTFPiK;
  G4FTFBinaryProtonBuilder  * theFTFPro;

  G4RPGNeutronBuilder       * theRPGNeutron;
  G4RPGPiKBuilder           * theRPGPiK;
  G4RPGProtonBuilder        * theRPGPro;

  G4String                    modelName;
  G4bool                      QuasiElastic;
};

#endif


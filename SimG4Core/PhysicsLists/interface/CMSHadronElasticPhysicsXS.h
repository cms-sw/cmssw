#ifndef SimG4Core_PhysicsLists_CMSHadronElasticPhysicsXS_h
#define SimG4Core_PhysicsLists_CMSHadronElasticPhysicsXS_h

#include "globals.hh"
#include "G4VPhysicsConstructor.hh"

class CMSHadronElasticPhysicsXS : public G4VPhysicsConstructor
{
public: 

  CMSHadronElasticPhysicsXS(G4int ver = 1); 

  virtual ~CMSHadronElasticPhysicsXS();

  // This method will be invoked in the Construct() method. 
  // each particle type will be instantiated
  virtual void ConstructParticle();
 
  // This method will be invoked in the Construct() method.
  // each physics process will be instantiated and
  // registered to the process manager of each particle type 
  virtual void ConstructProcess();

private:

  G4int    verbose;
  G4bool   wasActivated;
};


#endif









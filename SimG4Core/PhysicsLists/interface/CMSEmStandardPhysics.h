#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "G4MscStepLimitType.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CMSEmStandardPhysics : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysics(G4int ver, const edm::ParameterSet& p);
  ~CMSEmStandardPhysics() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4double fRangeFactor;
  G4double fGeomFactor;
  G4double fSafetyFactor;
  G4double fLambdaLimit;
  G4MscStepLimitType fStepLimitType;
  G4int verbose;
};

#endif

#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsEMH_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsEMH_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "G4MscStepLimitType.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CMSEmStandardPhysicsEMH : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsEMH(G4int ver, const edm::ParameterSet& p);
  ~CMSEmStandardPhysicsEMH() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4double fRangeFactor;
  G4double fGeomFactor;
  G4double fSafetyFactor;
  G4double fLambdaLimit;
  G4MscStepLimitType fStepLimitType;
};

#endif
